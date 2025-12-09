This program is an automated trading assistant for crypto.
It watches the market every minute, studies recent price behaviour, and decides whether it’s worth taking a short trade that aims to profit over the next 15 minutes.

You feed it historical market data (price, volume, indicators), and it uses a trained machine-learning model to estimate:

“How likely is the price to be higher 15 minutes from now?”

Based on that estimate, it follows a very simple rule:

- If the model is confident enough the price will go up (above  a set threshold) → it opens a small long position
- It holds the trade for about 15 minutes
- Then it exits the position, takes profit or loss, and waits for the next opportunity

It does not try to trade all the time. Most of the time it simply watches and does nothing, only acting when the data suggests the odds are slightly in its favour.

Key ideas, without the technical fluff
- It learns from past data to recognise short-term patterns
 It only trades when conditions look good enough, based on that learning
- It keeps trades short, aiming for quick, small moves (not long-term investing)
- It keeps everything structured and logged so you can later analyse:
   - how often it traded,
   - how often it was right,
   - and how much it would have made or lost.

Think of it as a cautious, stats-driven helper that suggests and executes quick trades when the odds look slightly better than a coin flip—then steps back and waits for the next setup.








