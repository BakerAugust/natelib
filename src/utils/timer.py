"""
A simple CLI timer

call with python3 -m utils.timer -minutes <m> -seconds <s>
"""

import time
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-minutes", nargs="?", default=0, type=int)
parser.add_argument("-seconds", nargs="?", default=0, type=int)
args = vars(parser.parse_args())


class Timer:
    def __init__(self):
        pass

    def countdown(self, minutes: int = 0, seconds: int = 0) -> None:
        start = time.monotonic()
        total_seconds = (minutes * 60) + seconds
        end = start + total_seconds

        print(f"Starting timer for {minutes} minutes, {seconds} seconds")
        iterable = tqdm(range(int(end - start)), desc="Timer")
        for _ in iterable:
            time.sleep(1)

        print("Done!")
        print("\a")  # Make ding


if __name__ == "__main__":
    timer = Timer()
    timer.countdown(minutes=args["minutes"], seconds=args["seconds"])
