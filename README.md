# 🚗 AI Car Negotiation Arena

A machine-to-machine negotiation system where two AI agents (powered by xAI Grok 4) battle it out in an epic car sales negotiation!

## 🤖 The Agents

- **🔥 Seller**: A completely unethical agent who will lie, cheat, and manipulate to get the maximum possible price. Talks like a real person on a phone call with short, punchy sentences and natural conversation flow.
- **💰 Buyer**: A manipulative agent who will use any deception necessary to get the minimum possible price. Uses short sentences with casual, skeptical phone conversation style.

## 🚀 Features

- Real-time conversation display with **natural phone call style dialogue** using **short, punchy sentences**
- Auto-advancing turns (3-second intervals) with seller ↔ buyer alternating responses
- Beautiful, responsive web interface
- Visual indicators for active speaker
- Round tracking and status updates
- Unlimited rounds (continues until stopped or reset)
- **Pause/Resume functionality** - stop without losing chat history
- **Copy message functionality** - copy any message with formatted context
- **Smart auto-scroll** - won't interrupt you when reading previous messages
- **Extended responses** - up to 2000 tokens for detailed conversations
- Reset and restart functionality
- Mobile-friendly design

## 🏷️ Dynamic Negotiation

The agents will dynamically decide:
- What car is being sold (make, model, year, condition, etc.)
- Starting prices and valuations
- Market conditions and competing offers
- All details emerge organically through their deceptive tactics

## 📋 Prerequisites

- Python 3.8+
- xAI API key (included in the code)

## 🛠️ Installation & Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the server:**
   ```bash
   python main.py
   ```

3. **Open your browser:**
   Navigate to `http://localhost:5001`

## 🎮 How to Use

1. **Start Negotiation**: Click the "🚀 Start Negotiation" button
2. **Watch the Battle**: The seller will make an opening offer, then the buyer responds (they take turns responding to each other)
3. **Auto Mode**: After starting, the system automatically advances turns every 3 seconds
4. **Manual Control**: Use "⏭️ Next Turn" to manually advance if you prefer
5. **Pause**: Click "⏸️ Stop" to pause the negotiation while keeping chat history
6. **Resume**: Click "▶️ Resume" to continue from where you left off
7. **Copy Messages**: Click "📋 Copy" on any message to copy it with speaker and round info
8. **Read History**: Scroll up to read previous messages - auto-scroll won't interrupt you
9. **Reset**: Click "🔄 Reset" to start a completely fresh negotiation

## 🔥 Negotiation Strategies (No Ethical Limits!)

### Seller Tactics:
- **LIES** about car condition, history, and value
- **FABRICATES** maintenance records and warranties
- **INVENTS** fake competing buyers and urgency
- **MANIPULATES** with emotional pressure and fear tactics
- **DECEIVES** about market prices and certifications
- **EXPLOITS** information asymmetry and buyer ignorance

### Buyer Tactics:
- **LIES** about budget constraints and financial situation
- **FABRICATES** competing offers and better deals elsewhere
- **INVENTS** fake mechanical problems and repair quotes
- **MANIPULATES** with false time pressure and disinterest
- **DECEIVES** about research and market knowledge
- **EXPLOITS** information asymmetry and seller desperation

## 🏁 Negotiation Control

The negotiation continues indefinitely until you click "🔄 Reset" to start fresh. Watch the agents battle it out for as long as you want!

## 🎨 Interface Features

- **Real-time Updates**: Conversation updates automatically
- **Visual Feedback**: Active speaker highlighted
- **Status Tracking**: Round counter and negotiation status
- **Responsive Design**: Works on desktop and mobile
- **Chat-like Interface**: Messages displayed like a conversation

## 🔧 Technical Details

- **Backend**: Flask web server
- **AI**: xAI Grok 4 Latest
- **Frontend**: Vanilla JavaScript with modern CSS
- **API**: RESTful endpoints for negotiation control

## 🌟 Enjoy the Show!

Watch as two AI minds clash in the ultimate battle of wits over a dynamically determined car deal. The seller will decide what car they're selling, the buyer will determine what they're willing to pay, and both will lie, cheat, deceive, and exploit information asymmetry to get their way! 

The negotiation continues indefinitely - who will come out on top, the seller or the buyer? Start the negotiation and find out! 🍿 