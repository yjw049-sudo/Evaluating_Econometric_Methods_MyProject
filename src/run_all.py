from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

SCRIPTS = [
    ROOT / "src" / "speech_merge.py",
    ROOT / "src" / "speech_clean.py",
    ROOT / "src" / "sample.py",
    ROOT / "src" / "speech_stem.py",
    ROOT / "src" / "speech_chunk.py",
    ROOT / "src" / "speech_stem_chunk.py",
    ROOT / "Kmeans" / "Kmeans_merged" / "run_kmeans.py",
    ROOT / "Kmeans" / "Kmeans_chunk" / "run_kmeans.py",
    ROOT / "Gemini_merged" / "06_merge_gemini_results_into_speech_sample.py",
    ROOT / "Gemini_chunk" / "06_merge_gemini_results_into_speech_chunk.py",
    ROOT / "src" / "merge_final.py",
    ROOT / "src" / "pic_merged.py",
    ROOT / "src" / "pic_chunk_check.py",
    ROOT / "src" / "merge_final.py",
    ROOT / "src" / "pic_main.py",
    ROOT / "src" / "pic_classification.py",
    ROOT / "src" / "pic_LLM.py",
    ROOT / "src" / "speech_descriptive.py"
]


def main():
    for script in SCRIPTS:
        print(f"\nRunning: {script.relative_to(ROOT)}")
        subprocess.run(
            [sys.executable, str(script)],
            cwd=ROOT,
            check=True,
        )

    print("\nAll scripts completed successfully.")


if __name__ == "__main__":
    main()