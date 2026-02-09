gt_file="ours_gt_file_keypoints.json"

pred_files=("example_pred.jsonl")


for pred_file in "${pred_files[@]}"; do
    echo "Running eval on: $pred_file"
    # calculate sodam
    python eval_sodam.py --pred_file "$pred_file" --gt_file $gt_file --max_workers 8

    # calculate merged F1 and mean IoU
    python eval_time.py --pred_file "$pred_file" --gt_file $gt_file --max_workers 8 --tmponly
done
