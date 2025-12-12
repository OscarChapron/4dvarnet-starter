#!/usr/bin/env python3
"""
Simple test script to validate configuration before running the full experiment.
"""
import subprocess
import sys

def test_config(xp_name, params_base, logger_name, ncomp, ignore_time, ignore_depth, dropout):
    """Test a single configuration"""
    cmd = [
        "python", "main.py",
        f"xp={xp_name}",
        f"+params={params_base}",
        f"+logger.name={logger_name}",
        f"datamodule.xrds_kw.patch_dims.component={ncomp}",
        f"prior_cost_unet_depth.ignore_time={ignore_time}",
        f"prior_cost_unet_depth.ignore_depth={ignore_depth}",
        f"prior_cost_unet_depth.dropout={dropout}",
        "trainer.max_epochs=1",
        "trainer.limit_train_batches=1",
        "trainer.limit_val_batches=1",
        "trainer.fast_dev_run=true"
    ]
    
    print(f"Testing config: ignore_time={ignore_time}, ignore_depth={ignore_depth}, dropout={dropout}, component={ncomp}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print("✅ Configuration test PASSED")
            return True
        else:
            print(f"❌ Configuration test FAILED with exit code {result.returncode}")
            print("STDERR:", result.stderr[-1000:])  # Last 1000 chars of stderr
            return False
    except subprocess.TimeoutExpired:
        print("❌ Configuration test TIMEOUT")
        return False
    except Exception as e:
        print(f"❌ Configuration test ERROR: {e}")
        return False

def main():
    """Test a few key configurations"""
    xp_name = "fdv_lazy_CTS_z"
    params_base = "direct_inversion_unet_z"
    
    # Test configurations from the script
    test_configs = [
        ("false", "false", "0.0", "1"),
        ("true", "false", "0.0", "1"),
        ("false", "true", "0.0", "1"),
        ("false", "false", "0.1", "1"),
    ]
    
    success_count = 0
    total_count = len(test_configs)
    
    for ignore_time, ignore_depth, dropout, ncomp in test_configs:
        logger_name = f"test_it_{ignore_time}_id_{ignore_depth}_do_{dropout}_c{ncomp}"
        
        if test_config(xp_name, params_base, logger_name, ncomp, ignore_time, ignore_depth, dropout):
            success_count += 1
        print("-" * 80)
    
    print(f"\nResults: {success_count}/{total_count} configurations passed")
    
    if success_count == total_count:
        print("🎉 All configurations work! The script should run successfully.")
        sys.exit(0)
    else:
        print("⚠️  Some configurations failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()