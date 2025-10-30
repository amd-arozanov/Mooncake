#!/bin/bash
# Script to run transfer_engine_validator tests with configurable parameters
# Usage: ./scripts/run_transfer_engine_validator.sh [MODE]
#   MODE can be: all (default), ipc, shareable
#   - all: Run both IPC and Shareable Handles tests
#   - ipc: Run only IPC (NVLink) tests
#   - shareable: Run only Shareable Handles tests

set -e  # Exit immediately if a command exits with a non-zero status
set -o pipefail  # Exit if any command in a pipeline fails

# Parse command line arguments
MODE="${1:-all}"

case "$MODE" in
    all|ipc|shareable)
        ;;
    *)
        echo "Usage: $0 [MODE]"
        echo "  MODE can be: all (default), ipc, shareable"
        echo "  - all: Run both IPC and Shareable Handles tests"
        echo "  - ipc: Run only IPC (NVLink) tests  "
        echo "  - shareable: Run only Shareable Handles tests"
        exit 1
        ;;
esac

# Configuration parameters - modify these as needed
BATCH_SIZES=(1 16)
BLOCK_SIZES=(1048576 2097152)
THREADS=(1 8 16)

# GPU configurations - arrays must have same length
TARGET_GPU_IDS=(0 0 -1)
WORKER_GPU_IDS=(2 0 -1)

DURATION=2

# Network configuration
METADATA_SERVER="127.0.0.1:2379"
TARGET_SERVER="127.0.0.2:14345"
WORKER_SERVER="127.0.0.3:14346"
ETCD_LISTEN_URL="http://0.0.0.0:2379"
ETCD_ADVERTISE_URL="http://10.0.0.1:2379"

# Build directory - adjust if your build is in a different location
BUILD_DIR="build"
VALIDATOR_PATH="${BUILD_DIR}/mooncake-transfer-engine/example/transfer_engine_validator"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to cleanup background processes
cleanup() {
    print_info "Cleaning up background processes..."
    if [ ! -z "$ETCD_PID" ]; then
        kill $ETCD_PID 2>/dev/null || true
        print_info "Stopped etcd (PID: $ETCD_PID)"
    fi
    if [ ! -z "$TARGET_PID" ]; then
        kill $TARGET_PID 2>/dev/null || true
        print_info "Stopped target (PID: $TARGET_PID)"
    fi
    # Kill any remaining transfer_engine_validator processes
    pkill -f transfer_engine_validator 2>/dev/null || true
    print_info "Cleanup completed"
}

# Set trap to cleanup on script exit
trap cleanup EXIT

# Check if validator executable exists
if [ ! -f "$VALIDATOR_PATH" ]; then
    print_error "transfer_engine_validator not found at $VALIDATOR_PATH"
    print_info "Please build the project first or adjust BUILD_DIR variable"
    exit 1
fi

# Check if etcd is available
if ! command -v etcd &> /dev/null; then
    print_error "etcd not found. Please install etcd first."
    exit 1
fi

# Validate GPU arrays have same length
if [ ${#TARGET_GPU_IDS[@]} -ne ${#WORKER_GPU_IDS[@]} ]; then
    print_error "TARGET_GPU_IDS and WORKER_GPU_IDS arrays must have the same length"
    print_error "TARGET_GPU_IDS has ${#TARGET_GPU_IDS[@]} elements: ${TARGET_GPU_IDS[*]}"
    print_error "WORKER_GPU_IDS has ${#WORKER_GPU_IDS[@]} elements: ${WORKER_GPU_IDS[*]}"
    exit 1
fi

print_info "Starting transfer_engine_validator test suite"
print_info "Test Mode: $MODE"
print_info "Configuration:"
print_info "  Batch sizes: ${BATCH_SIZES[*]}"
print_info "  Block sizes: ${BLOCK_SIZES[*]}"
print_info "  Thread counts: ${THREADS[*]}"
print_info "  GPU configurations:"
for i in "${!TARGET_GPU_IDS[@]}"; do
    target_gpu="${TARGET_GPU_IDS[i]}"
    worker_gpu="${WORKER_GPU_IDS[i]}"
    gpu_desc=""
    if [ "$target_gpu" = "-1" ] && [ "$worker_gpu" = "-1" ]; then
        gpu_desc=" (all GPUs)"
    elif [ "$target_gpu" = "-1" ]; then
        gpu_desc=" (target: all GPUs, worker: GPU $worker_gpu)"
    elif [ "$worker_gpu" = "-1" ]; then
        gpu_desc=" (target: GPU $target_gpu, worker: all GPUs)"
    else
        gpu_desc=" (target: GPU $target_gpu, worker: GPU $worker_gpu)"
    fi
    print_info "    Config $((i+1)): Target GPU $target_gpu, Worker GPU $worker_gpu$gpu_desc"
done

# Function to run test suite with specific mode
run_test_suite() {
    local mode_name="$1"
    local use_shareable_handles="$2"
    local env_vars="$3"
    
    print_info "=== Running $mode_name tests ==="
    
    # Loop through GPU configurations
    for gpu_config_idx in "${!TARGET_GPU_IDS[@]}"; do
        local target_gpu="${TARGET_GPU_IDS[gpu_config_idx]}"
        local worker_gpu="${WORKER_GPU_IDS[gpu_config_idx]}"
        
        local gpu_desc=""
        if [ "$target_gpu" = "-1" ] && [ "$worker_gpu" = "-1" ]; then
            gpu_desc="all GPUs"
        elif [ "$target_gpu" = "-1" ]; then
            gpu_desc="target: all GPUs, worker: GPU $worker_gpu"
        elif [ "$worker_gpu" = "-1" ]; then
            gpu_desc="target: GPU $target_gpu, worker: all GPUs"
        else
            gpu_desc="target: GPU $target_gpu, worker: GPU $worker_gpu"
        fi
        
        print_info "--- GPU Config $((gpu_config_idx+1)): $gpu_desc ---"
        
        # Start target with environment variables
        print_info "Starting target for $mode_name ($gpu_desc)..."
        if [ "$use_shareable_handles" = true ]; then
            eval "$env_vars" $VALIDATOR_PATH \
                --protocol=nvlink \
                --mode=target \
                --metadata_server=$METADATA_SERVER \
                --local_server_name=$TARGET_SERVER \
                --gpu_id=$target_gpu &
        else
            $VALIDATOR_PATH \
                --protocol=nvlink \
                --mode=target \
                --metadata_server=$METADATA_SERVER \
                --local_server_name=$TARGET_SERVER \
                --gpu_id=$target_gpu &
        fi
        TARGET_PID=$!
        print_info "Started target (PID: $TARGET_PID)"
        
        # Wait for target to be ready
        sleep 2
        
        # Run tests with different parameter combinations for this GPU config
        local gpu_test_count=0
        local gpu_successful_tests=0
        local gpu_failed_tests=0
        local gpu_validation_warnings=0
        
        for batch_size in "${BATCH_SIZES[@]}"; do
            for block_size in "${BLOCK_SIZES[@]}"; do
                for threads in "${THREADS[@]}"; do
                    test_count=$((test_count + 1))
                    gpu_test_count=$((gpu_test_count + 1))
                    
                    print_info "Running $mode_name test $gpu_test_count (GPU config $((gpu_config_idx+1))): batch_size=$batch_size, block_size=$block_size, threads=$threads"
                    
                    # Create temporary file for capturing output
                    test_output=$(mktemp)
                    
                    # Run worker and capture output with environment variables
                    if [ "$use_shareable_handles" = true ]; then
                        worker_cmd="eval \"$env_vars\" $VALIDATOR_PATH"
                    else
                        worker_cmd="$VALIDATOR_PATH"
                    fi
                    
                    # Execute the command and capture both output and exit code
                    set +e  # Temporarily disable exit on error to handle the exit code ourselves
                    eval "$worker_cmd" \
                        --protocol=nvlink \
                        --metadata_server=$METADATA_SERVER \
                        --segment_id=$TARGET_SERVER \
                        --local_server_name=$WORKER_SERVER \
                        --gpu_id=$worker_gpu \
                        -block_size=$block_size \
                        -batch_size=$batch_size \
                        -duration=$DURATION \
                        -threads=$threads 2>&1 | tee "$test_output"
                    exit_code=$?
                    set -e  # Re-enable exit on error
                    
                    if [ $exit_code -eq 0 ]; then
                        
                        # Check for data validation success and integrity problems
                        validation_passed=false
                        integrity_problem=false
                        
                        if grep -q "Data validation passed" "$test_output"; then
                            validation_passed=true
                        fi
                        
                        if grep -q "Detect data integrity problem" "$test_output"; then
                            integrity_problem=true
                        fi
                        
                        # Determine test result based on exit code and output validation
                        if [ "$validation_passed" = true ] && [ "$integrity_problem" = false ]; then
                            print_info "✓ $mode_name test $gpu_test_count completed successfully (data validation passed)"
                            successful_tests=$((successful_tests + 1))
                            gpu_successful_tests=$((gpu_successful_tests + 1))
                        elif [ "$integrity_problem" = true ]; then
                            print_error "✗ $mode_name test $gpu_test_count failed: data integrity problem detected"
                            failed_tests=$((failed_tests + 1))
                            gpu_failed_tests=$((gpu_failed_tests + 1))
                        elif [ "$validation_passed" = false ]; then
                            print_warning "⚠ $mode_name test $gpu_test_count completed but no 'Data validation passed' message found"
                            print_warning "  This might indicate the test didn't complete validation or output format changed"
                            validation_warnings=$((validation_warnings + 1))
                            gpu_validation_warnings=$((gpu_validation_warnings + 1))
                            failed_tests=$((failed_tests + 1))
                            gpu_failed_tests=$((gpu_failed_tests + 1))
                        else
                            print_info "✓ $mode_name test $gpu_test_count completed successfully"
                            successful_tests=$((successful_tests + 1))
                            gpu_successful_tests=$((gpu_successful_tests + 1))
                        fi
                    else
                        print_error "✗ $mode_name test $gpu_test_count failed with exit code $exit_code"
                        # Still check for integrity problems even if exit code is non-zero
                        if grep -q "Detect data integrity problem" "$test_output"; then
                            print_error "  Data integrity problem detected in output"
                        fi
                        failed_tests=$((failed_tests + 1))
                        gpu_failed_tests=$((gpu_failed_tests + 1))
                    fi
                    
                    # Clean up temporary file
                    rm -f "$test_output"
                    
                    # Small delay between tests
                    sleep 1
                done
            done
        done
        
        # Print GPU config summary
        print_info "$mode_name GPU Config $((gpu_config_idx+1)) Summary ($gpu_desc):"
        print_info "  Tests: $gpu_test_count"
        print_info "  Successful: $gpu_successful_tests"
        print_info "  Failed: $gpu_failed_tests"
        if [ $gpu_validation_warnings -gt 0 ]; then
            print_warning "  Validation warnings: $gpu_validation_warnings"
        fi
        
        # Stop target for this GPU config
        if [ ! -z "$TARGET_PID" ]; then
            kill $TARGET_PID 2>/dev/null || true
            print_info "Stopped target (PID: $TARGET_PID)"
            TARGET_PID=""
        fi
        
        # Small delay between GPU configs
        sleep 2
    done
}

# Start etcd
print_info "Starting etcd storage..."
etcd --listen-client-urls $ETCD_LISTEN_URL --advertise-client-urls $ETCD_ADVERTISE_URL &
ETCD_PID=$!
print_info "Started etcd (PID: $ETCD_PID)"

# Wait for etcd to be ready
sleep 2

# Initialize global counters
test_count=0
successful_tests=0
failed_tests=0
validation_warnings=0

# Run tests based on selected mode
case "$MODE" in
    all)
        # Run both IPC and Shareable Handles tests
        run_test_suite "IPC (NVLink)" false ""
        run_test_suite "Shareable Handles" true "MC_USE_NVLINK_IPC=0"
        ;;
    ipc)
        # Run only IPC tests
        run_test_suite "IPC (NVLink)" false ""
        ;;
    shareable)
        # Run only Shareable Handles tests
        run_test_suite "Shareable Handles" true "MC_USE_NVLINK_IPC=0"
        ;;
esac

# Print summary
print_info "=== Overall Test Summary ==="
print_info "  Total tests: $test_count"
print_info "  Successful: $successful_tests"
print_info "  Failed: $failed_tests"
if [ $validation_warnings -gt 0 ]; then
    print_warning "  Validation warnings: $validation_warnings"
fi

if [ $failed_tests -eq 0 ]; then
    print_info "All tests passed! 🎉"
    if [ $validation_warnings -gt 0 ]; then
        print_warning "Note: Some tests had validation warnings"
    fi
    exit 0
else
    print_error "Some tests failed!"
    exit 1
fi
