import json

def convert_dataset(dataset_path):
    with open(dataset_path, "r", encoding="utf-8") as f:
        data_samples = json.load(f)

    converted = []
    for sample in data_samples:
        for perspective in ["SUGGESTION", "CAUSE", "INFORMATION", "QUESTION", "EXPERIENCE"]:
            if perspective in sample.get("labelled_answer_spans", {}):
                spans = [entry["txt"] for entry in sample["labelled_answer_spans"][perspective]]
                summary_key = f"{perspective}_SUMMARY"
                converted.append({
                    "uri": sample.get("uri", ""),
                    "context": sample.get("context", ""),
                    "question": sample.get("question", ""),
                    "perspective": perspective,
                    "spans": spans,
                    "span_summary": sample.get("labelled_summaries", {}).get(summary_key, "")
                })

    output_path = f"InstructionSet/preprocessed_{dataset_path.split('/')[-1]}"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(converted, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    

    convert_dataset("Dataset/train.json")
    convert_dataset("Dataset/test.json")
    convert_dataset("Dataset/valid.json")

    with open(f"InstructionSet/preprocessed_train.json", "r", encoding="utf-8") as f:
        data_samples = json.load(f)
    print(data_samples[0])
