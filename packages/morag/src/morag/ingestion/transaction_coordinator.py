"""Transaction coordinator for atomic ingestion operations."""

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import structlog

from morag_core.config import DatabaseConfig, DatabaseType

logger = structlog.get_logger(__name__)


class TransactionState(Enum):
    """Transaction state enumeration."""
    PENDING = "pending"
    PREPARING = "preparing"
    PREPARED = "prepared"
    COMMITTING = "committing"
    COMMITTED = "committed"
    ABORTING = "aborting"
    ABORTED = "aborted"
    FAILED = "failed"


@dataclass
class TransactionOperation:
    """Represents a single operation within a transaction."""
    operation_id: str
    database_type: DatabaseType
    operation_type: str  # 'create_entities', 'create_relations', 'store_vectors', etc.
    data: Dict[str, Any]
    rollback_data: Optional[Dict[str, Any]] = None
    completed: bool = False
    error: Optional[str] = None


@dataclass
class IngestionTransaction:
    """Represents an atomic ingestion transaction."""
    transaction_id: str
    document_id: str
    source_path: str
    state: TransactionState = TransactionState.PENDING
    operations: List[TransactionOperation] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class TransactionCoordinator:
    """Coordinates atomic transactions across multiple databases."""
    
    def __init__(self):
        self.logger = logger.bind(component="transaction_coordinator")
        self.active_transactions: Dict[str, IngestionTransaction] = {}
        self.completed_transactions: Dict[str, IngestionTransaction] = {}
        self._lock = asyncio.Lock()
    
    async def begin_transaction(
        self,
        document_id: str,
        source_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Begin a new ingestion transaction.
        
        Args:
            document_id: Document identifier
            source_path: Source file path
            metadata: Optional transaction metadata
            
        Returns:
            Transaction ID
        """
        transaction_id = str(uuid.uuid4())
        
        transaction = IngestionTransaction(
            transaction_id=transaction_id,
            document_id=document_id,
            source_path=source_path,
            metadata=metadata or {}
        )
        
        async with self._lock:
            self.active_transactions[transaction_id] = transaction
        
        self.logger.info(
            "Transaction started",
            transaction_id=transaction_id,
            document_id=document_id,
            source_path=source_path
        )
        
        return transaction_id
    
    async def add_operation(
        self,
        transaction_id: str,
        database_type: DatabaseType,
        operation_type: str,
        data: Dict[str, Any],
        rollback_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add an operation to the transaction.
        
        Args:
            transaction_id: Transaction ID
            database_type: Target database type
            operation_type: Type of operation
            data: Operation data
            rollback_data: Data needed for rollback
            
        Returns:
            Operation ID
        """
        async with self._lock:
            if transaction_id not in self.active_transactions:
                raise ValueError(f"Transaction {transaction_id} not found")
            
            transaction = self.active_transactions[transaction_id]
            
            if transaction.state != TransactionState.PENDING:
                raise ValueError(f"Cannot add operations to transaction in state {transaction.state}")
            
            operation_id = str(uuid.uuid4())
            operation = TransactionOperation(
                operation_id=operation_id,
                database_type=database_type,
                operation_type=operation_type,
                data=data,
                rollback_data=rollback_data
            )
            
            transaction.operations.append(operation)
        
        self.logger.debug(
            "Operation added to transaction",
            transaction_id=transaction_id,
            operation_id=operation_id,
            database_type=database_type.value,
            operation_type=operation_type
        )
        
        return operation_id
    
    async def prepare_transaction(self, transaction_id: str) -> bool:
        """Prepare transaction for commit (validate all operations).
        
        Args:
            transaction_id: Transaction ID
            
        Returns:
            True if preparation successful, False otherwise
        """
        async with self._lock:
            if transaction_id not in self.active_transactions:
                raise ValueError(f"Transaction {transaction_id} not found")
            
            transaction = self.active_transactions[transaction_id]
            transaction.state = TransactionState.PREPARING
        
        try:
            self.logger.info(
                "Preparing transaction",
                transaction_id=transaction_id,
                operations_count=len(transaction.operations)
            )
            
            # Validate all operations
            for operation in transaction.operations:
                if not await self._validate_operation(operation):
                    transaction.state = TransactionState.FAILED
                    transaction.error_message = f"Operation validation failed: {operation.operation_id}"
                    return False
            
            async with self._lock:
                transaction.state = TransactionState.PREPARED
            
            self.logger.info("Transaction prepared successfully", transaction_id=transaction_id)
            return True
            
        except Exception as e:
            async with self._lock:
                transaction.state = TransactionState.FAILED
                transaction.error_message = str(e)
            
            self.logger.error(
                "Transaction preparation failed",
                transaction_id=transaction_id,
                error=str(e)
            )
            return False
    
    async def commit_transaction(self, transaction_id: str) -> bool:
        """Commit the transaction (execute all operations).
        
        Args:
            transaction_id: Transaction ID
            
        Returns:
            True if commit successful, False otherwise
        """
        async with self._lock:
            if transaction_id not in self.active_transactions:
                raise ValueError(f"Transaction {transaction_id} not found")
            
            transaction = self.active_transactions[transaction_id]
            
            if transaction.state != TransactionState.PREPARED:
                raise ValueError(f"Transaction must be prepared before commit. Current state: {transaction.state}")
            
            transaction.state = TransactionState.COMMITTING
        
        try:
            self.logger.info("Committing transaction", transaction_id=transaction_id)
            
            # Execute all operations
            for operation in transaction.operations:
                if not await self._execute_operation(operation):
                    # If any operation fails, abort the transaction
                    await self._abort_transaction_internal(transaction)
                    return False
                operation.completed = True
            
            # Mark transaction as committed
            async with self._lock:
                transaction.state = TransactionState.COMMITTED
                transaction.completed_at = datetime.now(timezone.utc)
                
                # Move to completed transactions
                self.completed_transactions[transaction_id] = transaction
                del self.active_transactions[transaction_id]
            
            self.logger.info("Transaction committed successfully", transaction_id=transaction_id)
            return True
            
        except Exception as e:
            await self._abort_transaction_internal(transaction)
            self.logger.error(
                "Transaction commit failed",
                transaction_id=transaction_id,
                error=str(e)
            )
            return False
    
    async def abort_transaction(self, transaction_id: str) -> bool:
        """Abort the transaction (rollback any completed operations).
        
        Args:
            transaction_id: Transaction ID
            
        Returns:
            True if abort successful, False otherwise
        """
        async with self._lock:
            if transaction_id not in self.active_transactions:
                raise ValueError(f"Transaction {transaction_id} not found")
            
            transaction = self.active_transactions[transaction_id]
        
        return await self._abort_transaction_internal(transaction)
    
    async def _abort_transaction_internal(self, transaction: IngestionTransaction) -> bool:
        """Internal method to abort a transaction."""
        try:
            async with self._lock:
                transaction.state = TransactionState.ABORTING
            
            self.logger.info("Aborting transaction", transaction_id=transaction.transaction_id)
            
            # Rollback completed operations in reverse order
            for operation in reversed(transaction.operations):
                if operation.completed:
                    await self._rollback_operation(operation)
            
            async with self._lock:
                transaction.state = TransactionState.ABORTED
                transaction.completed_at = datetime.now(timezone.utc)
                
                # Move to completed transactions
                self.completed_transactions[transaction.transaction_id] = transaction
                if transaction.transaction_id in self.active_transactions:
                    del self.active_transactions[transaction.transaction_id]
            
            self.logger.info("Transaction aborted successfully", transaction_id=transaction.transaction_id)
            return True
            
        except Exception as e:
            async with self._lock:
                transaction.state = TransactionState.FAILED
                transaction.error_message = str(e)
            
            self.logger.error(
                "Transaction abort failed",
                transaction_id=transaction.transaction_id,
                error=str(e)
            )
            return False
    
    async def get_transaction_status(self, transaction_id: str) -> Optional[IngestionTransaction]:
        """Get transaction status.
        
        Args:
            transaction_id: Transaction ID
            
        Returns:
            Transaction object or None if not found
        """
        async with self._lock:
            if transaction_id in self.active_transactions:
                return self.active_transactions[transaction_id]
            elif transaction_id in self.completed_transactions:
                return self.completed_transactions[transaction_id]
            return None
    
    async def cleanup_old_transactions(self, max_age_hours: int = 24) -> int:
        """Clean up old completed transactions.
        
        Args:
            max_age_hours: Maximum age in hours for completed transactions
            
        Returns:
            Number of transactions cleaned up
        """
        cutoff_time = datetime.now(timezone.utc).timestamp() - (max_age_hours * 3600)
        cleaned_count = 0
        
        async with self._lock:
            to_remove = []
            for transaction_id, transaction in self.completed_transactions.items():
                if transaction.completed_at and transaction.completed_at.timestamp() < cutoff_time:
                    to_remove.append(transaction_id)
            
            for transaction_id in to_remove:
                del self.completed_transactions[transaction_id]
                cleaned_count += 1
        
        if cleaned_count > 0:
            self.logger.info("Cleaned up old transactions", count=cleaned_count)
        
        return cleaned_count
    
    async def _validate_operation(self, operation: TransactionOperation) -> bool:
        """Validate an operation before execution."""
        # Basic validation - can be extended for specific operation types
        if not operation.data:
            return False
        
        # Validate required fields based on operation type
        if operation.operation_type == "create_entities":
            return "entities" in operation.data
        elif operation.operation_type == "create_relations":
            return "relations" in operation.data
        elif operation.operation_type == "store_vectors":
            return "vectors" in operation.data and "metadata" in operation.data
        
        return True
    
    async def _execute_operation(self, operation: TransactionOperation) -> bool:
        """Execute a single operation."""
        # This method will be implemented by the AtomicIngestionService
        # which has access to the actual database storage instances
        raise NotImplementedError("Operation execution must be implemented by the ingestion service")

    async def _rollback_operation(self, operation: TransactionOperation) -> bool:
        """Rollback a completed operation."""
        # This method will be implemented by the AtomicIngestionService
        # which has access to the actual database storage instances
        raise NotImplementedError("Operation rollback must be implemented by the ingestion service")


# Global transaction coordinator instance
_coordinator = TransactionCoordinator()

def get_transaction_coordinator() -> TransactionCoordinator:
    """Get the global transaction coordinator instance."""
    return _coordinator
