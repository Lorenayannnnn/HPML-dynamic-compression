import re

def clean_short_answer(short_answer: str):
    output = re.sub(r"(\d),(\d)", r"\1\2", short_answer)
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", output)
    if numbers:
        return numbers[-1]
    else:
        return output

def extract_gsm8k_answer(entry):
    short_answer = entry["answer"].split("####")[-1].strip()
    return clean_short_answer(short_answer)

def verify_gsm8k_answer(gt: str, prediction: str):
    from math_verify import parse, verify
    solution_str = parse(prediction)
    ground_truth = parse(gt)
    return 1 if verify(solution_str, ground_truth) else 0
