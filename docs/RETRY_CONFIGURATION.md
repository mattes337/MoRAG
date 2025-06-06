# Retry Configuration for Rate Limits

MoRAG now includes intelligent retry logic that handles rate limit errors gracefully with indefinite retries and exponential backoff.

## Overview

When processing large documents or batches of content, you may encounter rate limits from AI services like Google Gemini. Instead of failing after a few attempts, MoRAG can now retry indefinitely with intelligent backoff strategies.

## Key Features

- **Indefinite Retries**: Rate limit errors retry indefinitely (configurable)
- **Exponential Backoff**: Delays increase exponentially to reduce API pressure
- **Maximum Delay Cap**: Prevents excessive wait times (default: 5 minutes)
- **Jitter**: Random variation prevents thundering herd problems
- **Error-Specific Logic**: Only rate limits retry indefinitely; other errors use limited retries

## Configuration

### Environment Variables

Set these environment variables to customize retry behavior:

```bash
# Enable indefinite retries for rate limits (recommended)
MORAG_RETRY_INDEFINITELY=true

# Base delay between retries (seconds)
MORAG_RETRY_BASE_DELAY=1.0

# Maximum delay between retries (seconds) - prevents excessive waits
MORAG_RETRY_MAX_DELAY=300.0

# Exponential backoff multiplier
MORAG_RETRY_EXPONENTIAL_BASE=2.0

# Add random jitter to delays
MORAG_RETRY_JITTER=true
```

### Default Configuration

The default configuration is optimized for production use:

- **Indefinite retries**: Enabled
- **Base delay**: 1 second
- **Max delay**: 5 minutes (300 seconds)
- **Exponential base**: 2.0 (doubles each attempt)
- **Jitter**: Enabled

## Retry Behavior

### Rate Limit Errors

These error patterns trigger indefinite retries:
- HTTP 429 (Too Many Requests)
- "RESOURCE_EXHAUSTED"
- "quota exceeded"
- "rate limit"

**Example retry sequence:**
1. Attempt 1: Immediate
2. Attempt 2: Wait 1s
3. Attempt 3: Wait 2s
4. Attempt 4: Wait 4s
5. Attempt 5: Wait 8s
6. ...continues until success
7. Max delay: Capped at 300s

### Other Errors

Non-rate-limit errors (authentication, timeout, etc.) use limited retries (3 attempts) to prevent infinite loops.

## Example Configurations

### Aggressive Retrying (Fast Recovery)
```bash
MORAG_RETRY_INDEFINITELY=true
MORAG_RETRY_BASE_DELAY=0.5
MORAG_RETRY_MAX_DELAY=600.0
MORAG_RETRY_EXPONENTIAL_BASE=1.5
```

### Conservative Retrying (Slower, Less API Pressure)
```bash
MORAG_RETRY_INDEFINITELY=true
MORAG_RETRY_BASE_DELAY=2.0
MORAG_RETRY_MAX_DELAY=900.0
MORAG_RETRY_EXPONENTIAL_BASE=2.5
```

### Legacy Mode (Limited Retries)
```bash
MORAG_RETRY_INDEFINITELY=false
MORAG_RETRY_BASE_DELAY=1.0
```

## Testing

Test your retry configuration:

```bash
python scripts/test_retry_logic.py
```

This script will:
- Verify configuration loading
- Simulate exponential backoff delays
- Show retry behavior for different error types
- Display environment variable options

## Monitoring

When rate limits are hit, you'll see log messages like:

```
Rate limit hit, retrying indefinitely with exponential backoff
attempt=5, delay=16.2s, max_delay=300.0s
```

## Benefits

1. **Reliability**: Tasks no longer fail due to temporary rate limits
2. **Efficiency**: Exponential backoff reduces API pressure
3. **Scalability**: Handles high-volume processing gracefully
4. **Configurability**: Tune behavior for your specific needs

## Migration

Existing installations automatically get the new retry logic with safe defaults. No configuration changes are required, but you can customize the behavior using the environment variables above.

## Troubleshooting

### Tasks Taking Too Long

If tasks seem to hang, check the logs for retry messages. You may want to:
- Reduce `MORAG_RETRY_MAX_DELAY`
- Increase `MORAG_RETRY_BASE_DELAY` to be less aggressive

### Still Getting Rate Limit Failures

Ensure `MORAG_RETRY_INDEFINITELY=true` is set. Check that your API keys are valid and have sufficient quota.

### High API Usage

Consider:
- Increasing `MORAG_RETRY_BASE_DELAY`
- Reducing concurrent workers
- Processing smaller batches

## Technical Details

The retry logic is implemented in:
- `packages/morag-services/src/morag_services/embedding.py`
- `packages/morag-embedding/src/morag_embedding/service.py`
- `packages/morag-core/src/morag_core/config.py`

Rate limit detection uses pattern matching on error messages and HTTP status codes to distinguish between recoverable rate limits and permanent errors.
