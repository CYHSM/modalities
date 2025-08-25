#!/usr/bin/env python3
"""Simple script to test different temperatures with your model."""

from model_utils import generate_response, load_model_and_tokenizer

# Load your model
model_path = "/raid/s3/opengptx/mfrey/instruct/checkpoints/checkpoint-70000"  # Replace with your model path
model, tokenizer = load_model_and_tokenizer(
    model_path=model_path,
    device="cuda",
    torch_dtype="bfloat16",
    trust_remote_code=True,
    device_map="auto",
)

# Test prompt
prompt = """Solve the following math problem. Explain your reasoning and put the final answer in \boxed{}

Question: There are four schools competing at a basketball tournament. Each school has sent a girls’ basketball team and a boys’ basketball team and each team has 5 players each. Each school has also sent a coach for each team. In total, how many people have all of the schools sent?
# Answer:"""

# prompt = """Question: A certain store sells computer accessories and equipment. Due to a fire outbreak in one of the factories, the price of RAM increased by 30%. After two years, the price stabilized and finally fell by 20% from what it has risen. What is the current price of RAM if it was $50 before the fire?
# Answer: After the fire outbreak, the price of RAM got increased by 30/100 * 50 = $<<30/100*50=15>>15.
# So before stabilization the price was at 50 + 15 = $<<50+15=65>>65.
# After the stabilization, the price fell by 20% from $65, so it fell by 20/100 * 65 = $<<20/100*65=13>>13.
# That means the RAM is currently at 65 - 13 = $<<65-13=52>>52.
# #### 52

# Question: Last night, Jim bought a $7 lamp and a bulb which cost $4 less. If he bought 2 lamps and 6 bulbs, how much did Jim pay in all?
# Answer: The cost of the bulb is $7 - $4 = $<<7-4=3>>3.
# The cost of 2 lamps is $7 x 2 = $<<7*2=14>>14.
# The cost of 6 bulbs is $3 x 6 = $<<3*6=18>>18.
# Jim paid a total of $14 + $18 = $<<14+18=32>>32.
# #### 32

# Question: Joey needs to take a new prescription. The first day he needs to take one pill. Each day he must take two more pills than the previous day. How many pills will he take in a week?
# Answer: For the second day he takes 1 + 2 = <<1+2=3>>3 pills.
# For the third day he takes 3 + 2 = <<3+2=5>>5 pills.
# For the fourth day he takes 5 + 2 = <<5+2=7>>7 pills.
# For the fifth day he takes 7 + 2 = <<7+2=9>>9 pills.
# For the sixth day he takes 9 + 2 = <<9+2=11>>11 pills.
# For the seventh day he takes 11 + 2 = <<11+2=13>>13 pills.
# For the entire week he takes a total of 1 + 3 + 5 + 7 + 9 + 11 + 13 = <<1+3+5+7+9+11+13=49>>49 pills.
# #### 49

# Question: Sam bought a dozen boxes, each with 30 highlighter pens inside, for $10 each box. He rearranged five of these boxes into packages of six highlighters each and sold them for $3 per package. He sold the rest of the highlighters separately at the rate of three pens for $2. How much profit did he make in total, in dollars?
# Answer: Sam bought 12 boxes x $10 = $<<12*10=120>>120 worth of highlighters.
# He bought 12 * 30 = <<12*30=360>>360 highlighters in total.
# Sam then took 5 boxes × 6 highlighters/box = <<5*6=30>>30 highlighters.
# He sold these boxes for 5 * $3 = $<<5*3=15>>15
# After selling these 5 boxes there were 360 - 30 = <<360-30=330>>330 highlighters remaining.
# These form 330 / 3 = <<330/3=110>>110 groups of three pens.
# He sold each of these groups for $2 each, so made 110 * 2 = $<<110*2=220>>220 from them.
# In total, then, he earned $220 + $15 = $<<220+15=235>>235.
# Since his original cost was $120, he earned $235 - $120 = $<<235-120=115>>115 in profit.
# #### 115

# Question: Miranda is stuffing feather pillows. She needs two pounds of feathers for each pillow. A pound of goose feathers is approximately 300 feathers. Her goose has approximately 3600 feathers. How many pillows can she stuff after she plucks the goose?
# Answer: Miranda will have 3600 / 300 = <<3600/300=12>>12 pounds of feathers from her goose.
# Thus, Miranda can stuff about 12 / 2 = <<12/2=6>>6 feather pillows after she plucks the goose.
# #### 6

# Question: Peter and Andrew like to run in the morning.  Peter runs 3 miles more than Andrew's 2 miles.  After 5 days, how many miles have they both run?
# Answer: Peter runs 3 miles more than Andrew's 2 miles, so he runs 3+2 = <<3+2=5>>5 miles
# Peter runs 5 miles a day and Andrew runs 2 miles so they run 5+2 = <<5+2=7>>7 miles a day
# If they run 7 miles a day then over a course of 5 days they run 7*5 = <<7*5=35>>35 miles
# #### 35

# Question: Venus is at the deli to get subs for a party. She needs 81 inches of sub. The shop sells 5 and 8 inch subs. If she buys two 8 inch subs, how many 5 inch subs does she need to buy?
# Answer: The two 8 inch subs total 16 inches because 2 x 8 = <<2*8=16>>16
# She still needs 65 inches of sub because 81-16 = <<81-16=65>>65
# She needs to buy 13 five inch subs because 65 / 5 = <<65/5=13>>13
# #### 13

# Question: James buys $3000 worth of stuff from Amazon.  He has to return a TV that cost $700 and a bike that cost $500.  He also sells another bike that cost 20% more than the bike he returned for 80% of what he bought it for.  He then buys a toaster for $100.  How much is he out of pocket for everything?
# Answer: The items he returned were valued at $700 + $500 = $<<700+500=1200>>1200
# So far he is out 3000-1200 = <<3000-1200=1800>>1800 after recouping 1200.
# An item that is 20% more expensive cost 1 + .2 = 1.2 times as much as the item
# So that means the bike he sold cost $500 * 1.2 = $<<500*1.2=600>>600
# He sold it for $600 * .8 = $<<600*.8=480>>480
# From the bike that he had bought for 600, he was out 600-480 =<<600-480=120>>120
# So far he is out 1800+120 = <<1800+120=1920>>1920
# If he also bought a toaster worth 100, he was out 1920+100 = <<1920+100=2020>>2020
# #### 2020

# Question: There are four schools competing at a basketball tournament. Each school has sent a girls’ basketball team and a boys’ basketball team and each team has 5 players each. Each school has also sent a coach for each team. In total, how many people have all of the schools sent?
# Answer:"""

# Test different temperatures
temperatures = [0.1, 0.5, 0.7, 1.0, 1.2]

print(f"Prompt: {prompt}\n")
print("=" * 80)

for temp in temperatures:
    print(f"\n🌡️  Temperature: {temp}")
    print("-" * 40)

    # Generate 2 samples for each temperature to see variety
    for i in range(2):
        response = generate_response(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=200,
            temperature=temp,
            do_sample=True if temp > 0 else False,
            top_p=0.9,
        )
        print(f"Sample {i+1}: {response}\n")

print("=" * 80)
