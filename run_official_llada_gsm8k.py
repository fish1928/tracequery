import argparse
import inspect
import json
import os
import re
import sys

import torch


FEWSHOT_ROUNDS = [
    ("Question: Angelo and Melanie want to plan how many hours over the next week they should study together for their test next week. They have 2 chapters of their textbook to study and 4 worksheets to memorize. They figure out that they should dedicate 3 hours to each chapter of their textbook and 1.5 hours for each worksheet. If they plan to study no more than 4 hours each day, how many days should they plan to study total over the next week if they take a 10-minute break every hour, include 3 10-minute snack breaks each day, and 30 minutes for lunch each day?\nLet's think step by step\nAnswer:",
     "Angelo and Melanie think they should dedicate 3 hours to each of the 2 chapters, 3 hours x 2 chapters = 6 hours total.\nFor the worksheets they plan to dedicate 1.5 hours for each worksheet, 1.5 hours x 4 worksheets = 6 hours total.\nAngelo and Melanie need to start with planning 12 hours to study, at 4 hours a day, 12 / 4 = 3 days.\nHowever, they need to include time for breaks and lunch. Every hour they want to include a 10-minute break, so 12 total hours x 10 minutes = 120 extra minutes for breaks.\nThey also want to include 3 10-minute snack breaks, 3 x 10 minutes = 30 minutes.\nAnd they want to include 30 minutes for lunch each day, so 120 minutes for breaks + 30 minutes for snack breaks + 30 minutes for lunch = 180 minutes, or 180 / 60 minutes per hour = 3 extra hours.\nSo Angelo and Melanie want to plan 12 hours to study + 3 hours of breaks = 15 hours total.\nThey want to study no more than 4 hours each day, 15 hours / 4 hours each day = 3.75\nThey will need to plan to study 4 days to allow for all the time they need.\nThe answer is 4\n"),
    ("Question: Mark's basketball team scores 25 2 pointers, 8 3 pointers and 10 free throws.  Their opponents score double the 2 pointers but half the 3 pointers and free throws.  What's the total number of points scored by both teams added together?\nLet's think step by step\nAnswer:",
     "Mark's team scores 25 2 pointers, meaning they scored 25*2= 50 points in 2 pointers.\nHis team also scores 6 3 pointers, meaning they scored 8*3= 24 points in 3 pointers\nThey scored 10 free throws, and free throws count as one point so they scored 10*1=10 points in free throws.\nAll together his team scored 50+24+10= 84 points\nMark's opponents scored double his team's number of 2 pointers, meaning they scored 50*2=100 points in 2 pointers.\nHis opponents scored half his team's number of 3 pointers, meaning they scored 24/2= 12 points in 3 pointers.\nThey also scored half Mark's team's points in free throws, meaning they scored 10/2=5 points in free throws.\nAll together Mark's opponents scored 100+12+5=117 points\nThe total score for the game is both team's scores added together, so it is 84+117=201 points\nThe answer is 201\n"),
    ("Question: Bella has two times as many marbles as frisbees. She also has 20 more frisbees than deck cards. If she buys 2/5 times more of each item, what would be the total number of the items she will have if she currently has 60 marbles?\nLet's think step by step\nAnswer:",
     "When Bella buys 2/5 times more marbles, she'll have increased the number of marbles by 2/5*60 = 24\nThe total number of marbles she'll have is 60+24 = 84\nIf Bella currently has 60 marbles, and she has two times as many marbles as frisbees, she has 60/2 = 30 frisbees.\nIf Bella buys 2/5 times more frisbees, she'll have 2/5*30 = 12 more frisbees.\nThe total number of frisbees she'll have will increase to 30+12 = 42\nBella also has 20 more frisbees than deck cards, meaning she has 30-20 = 10 deck cards\nIf she buys 2/5 times more deck cards, she'll have 2/5*10 = 4 more deck cards.\nThe total number of deck cards she'll have is 10+4 = 14\nTogether, Bella will have a total of 14+42+84 = 140 items\nThe answer is 140\n"),
    ("Question: A group of 4 fruit baskets contains 9 apples, 15 oranges, and 14 bananas in the first three baskets and 2 less of each fruit in the fourth basket. How many fruits are there?\nLet's think step by step\nAnswer:",
     "For the first three baskets, the number of apples and oranges in one basket is 9+15=24\nIn total, together with bananas, the number of fruits in one basket is 24+14=38 for the first three baskets.\nSince there are three baskets each having 38 fruits, there are 3*38=114 fruits in the first three baskets.\nThe number of apples in the fourth basket is 9-2=7\nThere are also 15-2=13 oranges in the fourth basket\nThe combined number of oranges and apples in the fourth basket is 13+7=20\nThe fourth basket also contains 14-2=12 bananas.\nIn total, the fourth basket has 20+12=32 fruits.\nThe four baskets together have 32+114=146 fruits.\nThe answer is 146\n"),
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_llada', type=str, default='../LLaDA', help='path to the official LLaDA repo clone')
    parser.add_argument('--id_model', type=str, default='GSAI-ML/LLaDA-8B-Instruct')
    parser.add_argument('--limit', type=int, default=20)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--gen_length', type=int, default=256)
    parser.add_argument('--block_length', type=int, default=8)
    parser.add_argument('--steps', type=int, default=256)
    parser.add_argument('--num_fewshot', type=int, default=4, help='0..4 exemplars from the official template')
    parser.add_argument('--path_output', type=str, default='official_llada_gsm8k.json')
    return parser.parse_args()


def build_messages(question, num_fewshot):
    messages = []
    for human, bot in FEWSHOT_ROUNDS[:num_fewshot]:
        messages.append({'role': 'user', 'content': human})
        messages.append({'role': 'assistant', 'content': bot})
    messages.append({'role': 'user', 'content': f"Question: {question}\nLet's think step by step\nAnswer:"})
    return messages


def extract_last_number(text):
    text = text.split('Question:')[0]
    numbers = re.findall(r'-?[0-9][0-9,]*\.?[0-9]*', text.replace('$', ''))
    if not numbers:
        return None
    return numbers[-1].replace(',', '').rstrip('.')


def extract_gold(answer):
    return answer.split('####')[-1].strip().replace(',', '')


def main():
    args = parse_args()

    sys.path.insert(0, os.path.abspath(args.path_llada))
    from generate import generate as llada_generate

    from datasets import load_dataset
    from transformers import AutoModel, AutoTokenizer

    ds = load_dataset('gsm8k', 'main', split='test').select(range(args.limit))

    tokenizer = AutoTokenizer.from_pretrained(args.id_model, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.id_model, trust_remote_code=True, torch_dtype=torch.bfloat16)
    model = model.eval().to(args.device)

    params_generate = set(inspect.signature(llada_generate).parameters)
    kwargs_extra = {}
    for name in ('logits_eos_inf', 'confidence_eos_eot_inf'):
        if name in params_generate:
            kwargs_extra[name] = False

    rows = []
    num_correct = 0
    for id_sample, sample in enumerate(ds):
        messages = build_messages(sample['question'], args.num_fewshot)
        text_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        ids_prompt = tokenizer.batch_encode_plus([text_prompt], padding=True, return_tensors='pt')['input_ids']

        with torch.no_grad():
            x = llada_generate(
                model=model,
                prompt=ids_prompt.to(args.device),
                steps=args.steps,
                gen_length=args.gen_length,
                block_length=args.block_length,
                temperature=0.,
                cfg_scale=0.,
                remasking='low_confidence',
                mask_id=126336,
                **kwargs_extra,
            )

        text_generated = tokenizer.decode(x[0, -args.gen_length:], skip_special_tokens=True)
        prediction = extract_last_number(text_generated)
        gold = extract_gold(sample['answer'])
        correct = prediction == gold
        num_correct += int(correct)

        print(f"[{id_sample}] gold={gold} pred={prediction} correct={correct} "
              f"running_acc={num_correct / (id_sample + 1):.3f}")
        print(f"    {text_generated[:300]!r}")

        rows.append({
            'id_sample': id_sample,
            'gold': gold,
            'prediction': prediction,
            'correct': correct,
            'text_generated': text_generated,
        })

    accuracy = num_correct / len(rows)
    print(f"\nOFFICIAL LLaDA pipeline: {num_correct}/{len(rows)} = {accuracy:.3f} "
          f"(gen {args.gen_length}, block {args.block_length}, {args.num_fewshot}-shot CoT)")

    with open(args.path_output, 'w') as file:
        json.dump({
            'id_model': args.id_model,
            'gen_length': args.gen_length,
            'block_length': args.block_length,
            'steps': args.steps,
            'num_fewshot': args.num_fewshot,
            'accuracy': accuracy,
            'rows': rows,
        }, file, indent=2)


if __name__ == '__main__':
    main()
