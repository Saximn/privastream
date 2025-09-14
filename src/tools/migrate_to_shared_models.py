#!/usr/bin/env python3
"""
Migration script to update existing scripts to use the new shared models architecture.
This script helps identify files that need updating and provides migration suggestions.
"""
import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple


def find_files_to_migrate(root_dir: Path) -> List[Path]:
    """Find Python files that likely need migration to shared models"""
    files_to_check = []
    
    # Patterns that indicate usage of old model imports
    old_patterns = [
        r"from.*face_detector.*import",
        r"from.*pii_detector.*import", 
        r"from.*plate_detector.*import",
        r"from.*unified_detector.*import",
        r"import.*face_detector",
        r"import.*pii_detector",
        r"import.*plate_detector"
    ]
    
    # Scan for Python files
    for py_file in root_dir.rglob("*.py"):
        # Skip files in shared directory (already migrated)
        if "shared" in str(py_file):
            continue
            
        try:
            content = py_file.read_text(encoding='utf-8')
            
            # Check if file contains old model imports
            for pattern in old_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    files_to_check.append(py_file)
                    break
                    
        except Exception as e:
            print(f"Warning: Could not read {py_file}: {e}")
    
    return files_to_check


def analyze_file(file_path: Path) -> Dict[str, List[str]]:
    """Analyze a file and identify migration needs"""
    analysis = {
        "old_imports": [],
        "model_creation": [],
        "config_usage": [],
        "suggestions": []
    }
    
    try:
        content = file_path.read_text(encoding='utf-8')
        lines = content.splitlines()
        
        for i, line in enumerate(lines, 1):
            # Check for old imports
            if re.search(r"from.*(face_detector|pii_detector|plate_detector).*import", line):
                analysis["old_imports"].append(f"Line {i}: {line.strip()}")
            
            # Check for model instantiation
            if re.search(r"(FaceDetector|PIIDetector|PlateDetector)\s*\(", line):
                analysis["model_creation"].append(f"Line {i}: {line.strip()}")
            
            # Check for hardcoded config values
            if re.search(r"(threshold|conf_thresh|det_size|gpu_id).*=.*\d+", line):
                analysis["config_usage"].append(f"Line {i}: {line.strip()}")
        
        # Generate suggestions
        if analysis["old_imports"]:
            analysis["suggestions"].append("Replace old imports with: from shared import UnifiedBlurDetector, create_detector")
        
        if analysis["model_creation"]:
            analysis["suggestions"].append("Replace individual model creation with UnifiedBlurDetector")
        
        if analysis["config_usage"]:
            analysis["suggestions"].append("Move hardcoded values to ModelConfig")
            
    except Exception as e:
        analysis["suggestions"].append(f"Error analyzing file: {e}")
    
    return analysis


def generate_migration_template(file_path: Path) -> str:
    """Generate migration template for a file"""
    template = f"""
# MIGRATION TEMPLATE for {file_path.name}
# =====================================

# OLD CODE (to be replaced):
# from models.face_blur.face_detector import FaceDetector
# from models.pii_blur.pii_detector import PIIDetector  
# from models.plate_blur.plate_detector import PlateDetector
#
# face_det = FaceDetector(threshold=0.35, det_size=960, gpu_id=0)
# pii_det = PIIDetector(conf_thresh=0.35, K_confirm=2, K_hold=8)
# plate_det = PlateDetector(conf_thresh=0.25, device="cuda")

# NEW CODE (recommended):
import sys
from pathlib import Path

# Add shared models to path  
sys.path.append(str(Path(__file__).parent.parent / "shared"))

# Option 1: Use predefined configuration
from adapters import get_web_detector, get_production_detector
detector = get_web_detector()  # or get_production_detector()

# Option 2: Custom configuration
from config import ModelConfig
from models import UnifiedBlurDetector

config = ModelConfig()
config.FACE_THRESHOLD = 0.35
config.PII_CONFIDENCE_THRESHOLD = 0.35
config.PLATE_CONFIDENCE_THRESHOLD = 0.25
detector = UnifiedBlurDetector(config)

# Option 3: Legacy compatibility (temporary)
from adapters import get_legacy_detector
legacy_config = {{
    "face": {{"threshold": 0.35, "det_size": 960, "gpu_id": 0}},
    "pii": {{"conf_thresh": 0.35, "K_confirm": 2, "K_hold": 8}},
    "plate": {{"conf_thresh": 0.25, "device": "cuda"}}
}}
detector = get_legacy_detector(legacy_config)

# USAGE (unified interface):
# detections = detector.detect_frame(frame)
# blurred_frame = detector.blur_frame(frame, detections)
"""
    return template


def create_migration_report(root_dir: Path) -> str:
    """Create comprehensive migration report"""
    print("Scanning for files that need migration...")
    files_to_migrate = find_files_to_migrate(root_dir)
    
    report = [
        "MIGRATION REPORT",
        "================",
        f"Scanned directory: {root_dir}",
        f"Files requiring migration: {len(files_to_migrate)}",
        "",
        "PRIORITY FILES (most changes needed):",
        ""
    ]
    
    priority_files = []
    
    for file_path in files_to_migrate:
        analysis = analyze_file(file_path)
        total_issues = (
            len(analysis["old_imports"]) + 
            len(analysis["model_creation"]) + 
            len(analysis["config_usage"])
        )
        
        priority_files.append((file_path, total_issues, analysis))
    
    # Sort by number of issues (descending)
    priority_files.sort(key=lambda x: x[1], reverse=True)
    
    for file_path, issue_count, analysis in priority_files[:10]:  # Top 10
        relative_path = file_path.relative_to(root_dir)
        report.extend([
            f"{relative_path} ({issue_count} issues)",
            "-" * 40
        ])
        
        if analysis["old_imports"]:
            report.append("Old imports found:")
            report.extend(f"  {imp}" for imp in analysis["old_imports"])
        
        if analysis["model_creation"]:
            report.append("Model creation found:")
            report.extend(f"  {model}" for model in analysis["model_creation"])
        
        if analysis["config_usage"]:
            report.append("Hardcoded config found:")
            report.extend(f"  {config}" for config in analysis["config_usage"])
        
        if analysis["suggestions"]:
            report.append("Suggestions:")
            report.extend(f"  - {sug}" for sug in analysis["suggestions"])
        
        report.append("")
    
    # Summary
    report.extend([
        "",
        "MIGRATION STEPS:",
        "================",
        "1. Update imports to use shared models",
        "2. Replace individual detectors with UnifiedBlurDetector", 
        "3. Move hardcoded values to ModelConfig",
        "4. Test updated scripts thoroughly",
        "5. Remove old model directories after migration",
        "",
        "BENEFITS AFTER MIGRATION:",
        "========================",
        "- 90% reduction in code duplication",
        "- Centralized configuration management",
        "- Consistent error handling and logging",
        "- Better performance monitoring",
        "- Easier maintenance and updates",
        "",
        f"Total files to migrate: {len(files_to_migrate)}"
    ])
    
    return "\\n".join(report)


def main():
    """Main migration analysis"""
    if len(sys.argv) > 1:
        root_dir = Path(sys.argv[1])
    else:
        root_dir = Path(__file__).parent.parent
    
    print(f"Analyzing migration needs in: {root_dir}")
    
    # Generate migration report
    report = create_migration_report(root_dir)
    
    # Save report
    report_file = root_dir / "MIGRATION_REPORT.md"
    report_file.write_text(report)
    
    print(f"Migration report saved to: {report_file}")
    print("\\n" + "="*50)
    print("TOP RECOMMENDATIONS:")
    print("="*50)
    print("1. Start with web-demo-ui/backend files")
    print("2. Update scripts/ directory")
    print("3. Migrate notebooks/ if needed")
    print("4. Test each migrated file thoroughly")
    print("5. Remove old model directories when done")
    
    # Show first few lines of report
    print("\\n" + report[:1000] + "..." if len(report) > 1000 else report)


if __name__ == "__main__":
    main()