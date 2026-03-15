# AlphaScout — 3-Minute Pitch Video Script

> Total duration: 3:00
> Format: Founder talking to camera + screen recording intercut
> Tone: Confident, conversational, not salesy. Like explaining to a smart friend.

---

## PART 1: HOOK (0:00 – 0:20)

### Script

> "You want to use AI to trade stocks. So you Google it. And you find three options."
>
> "Bloomberg Terminal — twenty-four thousand dollars a year. QuantConnect — great, but you need to write Python. TradingView — no machine learning at all, just old-school indicators."
>
> "What if I told you there's a better way?"

### Visual

- **0:00–0:05** — Founder facing camera, dark background, confident opening
- **0:05–0:08** — Quick flash: Bloomberg Terminal screenshot, price tag "$24,000/yr" overlaid in bold red
- **0:08–0:11** — Quick flash: QuantConnect IDE with Python code, text overlay "Requires coding"
- **0:11–0:14** — Quick flash: TradingView chart with RSI/MACD, text overlay "No ML"
- **0:14–0:20** — Back to founder, slight lean forward, "What if I told you..."

---

## PART 2: THE INSIGHT (0:20 – 0:50)

### Script

> "Here's something most people don't know. Different stocks need different AI models."
>
> "Apple doesn't trade like Tesla. A Transformer might crush it on NVDA, but completely fail on MSFT. The only way to know which AI model works best for YOUR stock — is to test them all."
>
> "That's exactly what AlphaScout does."
>
> "You type a stock ticker. We run twenty-two machine learning models on that stock. The best one wins. And you get a simple answer: buy, sell, or hold."

### Visual

- **0:20–0:28** — Founder speaking, animated split-screen appears: left shows AAPL chart, right shows TSLA chart — clearly different patterns
- **0:28–0:35** — Simple animation: two model names (e.g., "Transformer" and "LSTM") swap rankings on AAPL vs TSLA — demonstrating that the best model changes per stock
- **0:35–0:40** — Text animation on screen: "22 AI Models × Your Stock = The Best Strategy"
- **0:40–0:50** — Founder speaking to camera, text overlay appears with the three steps: "Type a ticker → 22 models compete → Get your answer"

---

## PART 3: LIVE DEMO (0:50 – 1:50)

### Script

> "Let me show you. This is live."
>
> *(switches to screen recording)*
>
> "I go to AlphaScout. I type 'AAPL'. I hit 'Get AlphaScore'."
>
> *(AlphaScore ring animates)*
>
> "There it is. That score comes from twenty-two models that all trained on Apple's real market data. Let me show you the details."
>
> *(clicks "View Full Analysis", switches to app page, shows scan results table)*
>
> "Here's every model, ranked. You can see: GRU returned plus twelve percent. But Apple only went up six percent in the same period. So GRU beat buy-and-hold by six points. That's real alpha."
>
> *(scrolls to show different models with different returns)*
>
> "And this is the key part — these are out-of-sample results. The model never saw this test data during training. This isn't curve-fitting. This is real predictive performance."
>
> *(points to training period banner showing train/valid/test dates)*
>
> "Now, here's the thing — that full scan takes time. So we built Quick Predict. I click here, type any ticker, and get an instant signal from a pre-trained model. Buy, sell, or hold — in seconds, not minutes."
>
> *(shows Quick Predict tab, types a ticker, gets instant result)*

### Visual

- **0:50–0:52** — Founder gestures to screen, transitions to screen recording
- **0:52–1:00** — Landing page hero section, cursor types "AAPL" in search box, clicks "Get AlphaScore"
- **1:00–1:08** — AlphaScore ring animates, score and signal appear. Pause for 2 seconds to let it sink in.
- **1:08–1:12** — Click "View Full Analysis" button, transition to /app page
- **1:12–1:25** — Scan results table visible. Zoom/highlight on the #1 ranked model row: model return, buy-and-hold return, excess return column. Use a cursor circle or highlight box.
- **1:25–1:35** — Scroll down through the full table showing all 22 models ranked, some green (profitable), some red
- **1:35–1:40** — Zoom into the training period banner: "Train 6 months → Validate 1 month → Test 1 month" with the exact date ranges. Highlight "Test period (out-of-sample)" text.
- **1:40–1:50** — Click "Quick Predict" tab, type "NVDA", instant signal appears (buy/sell/hold card + score). Highlight the speed — result in 3-5 seconds.

---

## PART 4: WHY THIS MATTERS — MARKET (1:50 – 2:10)

### Script

> *(back to founder)*
>
> "Since 2020, over a hundred and fifty million new people started trading stocks. They're all looking for an edge. But right now, their options are either too expensive, too technical, or too simplistic."
>
> "The retail trading tools market is twelve billion dollars. AI-powered trading is the fastest-growing segment. And automatic multi-model comparison? Nobody does this. We're the only one."
>
> "Danelfin does AI scores but uses one model. TipRanks aggregates analyst opinions. We actually train twenty-two models on your specific stock and let them compete."

### Visual

- **1:50–1:58** — Founder speaking. Behind: animated counter "150M+ new retail traders since 2020" ticking up
- **1:58–2:04** — Market size visualization: three bars — $12B TAM, $3.2B SAM, $180M SOM. Simple, clean, dark background.
- **2:04–2:10** — Quick comparison: AlphaScout logo vs Danelfin / TipRanks / TradingView. Each with a one-line differentiator: "1 model" / "Analyst opinions" / "No ML" vs "22 models compete"

---

## PART 5: BUSINESS MODEL + TECH (2:10 – 2:35)

### Script

> "Right now, AlphaScout is completely free — I'm running it on my own GPU server so you can try it with zero cost."
>
> "We're built on Microsoft Qlib — the same quantitative finance engine that institutional quants use — running on an NVIDIA RTX PRO 6000 GPU."
>
> "To scale, we'll move to cloud GPU clusters and offer a membership with credits. Free users get a few scans per month. Members pay a monthly fee, get a credit allowance, and each scan costs credits — because every model run burns real GPU time. It's usage-based pricing that maps directly to our cost structure."
>
> "Next quarter: mobile app with push notifications. After that: brokerage integration for one-click trading."

### Visual

- **2:10–2:15** — Browser showing the live product at alphascout.bjjwwangs.win. Cursor clicks around to show it's real and responsive.
- **2:15–2:22** — Tech stack visual: Microsoft Qlib logo → PyTorch logo → NVIDIA logo. Clean, minimal. Architecture flow: Data → 158 Features → 22 Models → AlphaScore → Signal
- **2:22–2:30** — Three pricing cards: Free (3 credits/mo), Pro ($19/mo, 50 credits), Power ($49/mo, 200 credits). The Pro card is highlighted/larger. Text below: "1 Quick Predict = 1 credit, 1 Full Scan = 5 credits". Keep it on screen for 3-4 seconds.
- **2:30–2:35** — Roadmap timeline: Q2 2026 "Mobile App" → Q3 "Brokerage Integration" → Q4 "API Marketplace". Simple horizontal timeline.

---

## PART 6: CLOSE — CALL TO ACTION (2:35 – 3:00)

### Script

> "Let me leave you with this."
>
> "Ninety-five percent of retail traders lose money. Not because they're dumb — because they don't have the right tools. We built the tool."
>
> "Twenty-two AI models. Over six hundred US stocks. One clear answer."
>
> *(pause, look at camera)*
>
> "Try it right now. alphascout.bjjwwangs.win. Type any stock. Get your AlphaScore. And decide: should you buy, sell, or hold."
>
> *(slight smile)*
>
> "Thank you."

### Visual

- **2:35–2:42** — Founder speaking, serious but not dramatic. Dark background, good lighting.
- **2:42–2:48** — Text animation on screen: "22 AI Models. 600+ US Stocks. One Answer." Each phrase appears one by one with a slight fade.
- **2:48–2:55** — Landing page hero with search box, cursor typing a ticker, AlphaScore ring appearing. URL "alphascout.bjjwwangs.win" prominently displayed below.
- **2:55–3:00** — Back to founder. Slight nod. "Thank you." Fade to AlphaScout logo + URL on dark background.

---

## Production Notes

### Recording Tips

1. **Founder shots**: Use a fixed camera, dark/clean background (wall or curtain), warm side lighting. Don't stand behind a desk — chest up, centered.
2. **Screen recordings**: Use a clean browser window (no bookmarks bar, no extensions visible). Dark mode. Maximize the window. Record at 1080p or higher.
3. **Transitions**: Use simple cuts, not fancy transitions. Speed is more credible than polish.
4. **Audio**: Record voiceover separately if the room is noisy. A lapel mic or AirPods Pro is fine. No background music during the demo section (it distracts).
5. **Pacing**: The script is tight. Practice it 3-4 times. If you're running over 3:00, cut the business model section shorter (it's the least important for the competition).

### Key Moments to Nail

- **0:00–0:05**: First impression. Be confident, not nervous. One sentence, look straight into the camera.
- **1:00–1:08**: The AlphaScore ring animation. This is the "wow" moment. Let it breathe for 2 seconds after it appears.
- **1:35–1:40**: Out-of-sample explanation. This is where you earn trust. Slow down here.
- **1:40–1:50**: Quick Predict demo. This shows speed and practical value. The instant result is the second "wow" moment.
- **2:48–2:55**: The live URL. Make sure it's readable. This is what judges will type in right after watching.

### What Judges Are Looking For

Based on UNSW Founders criteria:
- **Problem/Solution clarity**: Is the problem real? Does the solution make sense? (Part 1 + 2)
- **Demo/traction**: Is this real or just a deck? (Part 3 — the live demo is your strongest card)
- **Market opportunity**: Is this a big enough market? (Part 4)
- **Business viability**: Can this make money? (Part 5)
- **Team**: Can this person execute? (Your confidence throughout)
