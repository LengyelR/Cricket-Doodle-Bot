# Cricket-Doodle-Bot

Learning off-policy, just by watching a simple bot play. Using a policy gradient algorithm, it can learn to play the game in a few hours.


<img src="https://github.com/LengyelR/Cricket-Doodle-Bot/blob/master/highscore.jpg" width="640">


## SAR
 - State: 5 consecutive frames
 - Reward: change in the last digit of the scoreboard
 - Action: clicking or idling

## Getting rewards
I created a simple bot, which analysed the screenshots and if it detected the ball in one, it hit it.
Unfortunately the PIL ImageGrab module was not fast enough for this task.
After resorting to native handles, I managed to make it ~10x faster, which was more than enough.

While the bot was playing, it was also monitoring the scoreboard. Due to the moving board, the collected data was noisy at first. After a few improvements the bot was creating and saving the cleaned images.

I created a GUI that showed 3 images at the same time, and just by pressing the number keys, the app labeled them. I was able to label the whole dataset in matter of minutes.

The images were fairly similar, a simple k-nearest neighbours algorithm would probably suffice to classify them, but I was already over-engineering it, so conv nets it is.
After training on 1000 images and cross-validating on the rest, it achieved a 100% accuracy.

## Everything in action
Now the RL framework is ready. All left is to train a new model to play the game.
The training happens on another thread (overcoming the GIL), so the main thread can still collect the SAR values.
