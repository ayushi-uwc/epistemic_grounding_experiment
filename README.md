# 🧠 Epistemic Grounding Experiment

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-2.3+-green.svg)](https://flask.palletsprojects.com/)

[![GitHub stars](https://img.shields.io/github/stars/ayushi-uwc/epistemic_grounding_experiment?style=social)](https://github.com/ayushi-uwc/epistemic_grounding_experiment/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/ayushi-uwc/epistemic_grounding_experiment?style=social)](https://github.com/ayushi-uwc/epistemic_grounding_experiment/network/members)
[![GitHub issues](https://img.shields.io/github/issues/ayushi-uwc/epistemic_grounding_experiment)](https://github.com/ayushi-uwc/epistemic_grounding_experiment/issues)
[![GitHub pull requests](https://img.shields.io/github/issues-pr/ayushi-uwc/epistemic_grounding_experiment)](https://github.com/ayushi-uwc/epistemic_grounding_experiment/pulls)

[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4%20%7C%20O1%20%7C%20O3-412991?logo=openai)](https://openai.com/)
[![Anthropic](https://img.shields.io/badge/Anthropic-Claude%204-orange?logo=anthropic)](https://anthropic.com/)
[![xAI](https://img.shields.io/badge/xAI-Grok-black?logo=x)](https://x.ai/)
[![Groq](https://img.shields.io/badge/Groq-Llama%20%7C%20Mixtral-brightgreen)](https://groq.com/)

[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey)](https://github.com/ayushi-uwc/epistemic_grounding_experiment)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/ayushi-uwc/epistemic_grounding_experiment/blob/main/CONTRIBUTING.md)
[![Research](https://img.shields.io/badge/research-AI%20%7C%20ML%20%7C%20NLP-blue?logo=academia)](https://github.com/ayushi-uwc/epistemic_grounding_experiment)

[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/ayushi-uwc/epistemic_grounding_experiment/graphs/commit-activity)
[![GitHub last commit](https://img.shields.io/github/last-commit/ayushi-uwc/epistemic_grounding_experiment)](https://github.com/ayushi-uwc/epistemic_grounding_experiment/commits/main)
[![GitHub repo size](https://img.shields.io/github/repo-size/ayushi-uwc/epistemic_grounding_experiment)](https://github.com/ayushi-uwc/epistemic_grounding_experiment)
[![Lines of code](https://img.shields.io/tokei/lines/github/ayushi-uwc/epistemic_grounding_experiment)](https://github.com/ayushi-uwc/epistemic_grounding_experiment)

A cutting-edge research platform for studying **epistemic grounding** and **AI-to-AI negotiations** through machine learning conversations. This platform enables researchers to explore how artificial agents communicate, negotiate, and establish shared understanding.

## 🌟 Overview

The Epistemic Grounding Experiment is a comprehensive platform that facilitates machine-to-machine negotiations across multiple AI providers. It provides tools for:

- **AI-to-AI Negotiations**: Real-time conversations between different AI agents
- **Multi-Provider Support**: Integration with OpenAI, Anthropic, xAI, and Groq
- **Batch Experimentation**: Large-scale automated negotiation studies
- **Data Analysis**: Comprehensive conversation and outcome analysis
- **Web Interface**: Intuitive browser-based control and monitoring

## 🎯 Research Applications

- **Epistemic Modeling**: Study how AI agents establish and maintain shared knowledge
- **Negotiation Dynamics**: Analyze strategies and outcomes in AI-to-AI bargaining
- **Communication Patterns**: Explore how different AI models interact and communicate
- **Multi-Agent Systems**: Research coordination and competition between AI agents
- **Language Evolution**: Investigate how communication protocols emerge between agents

## ✨ Key Features

### 🚀 Multi-LLM Support
- **OpenAI**: GPT-4, GPT-4o, O1, O3 series
- **Anthropic**: Claude Sonnet 4, Claude 3.7 Sonnet  
- **xAI**: Grok-4, Grok-3 series
- **Groq**: Llama, Mixtral, Gemma models

### 🎮 Interactive Platform
- **Real-time Negotiations**: Watch AI agents negotiate live
- **Batch Processing**: Run multiple experiments simultaneously
- **Custom Scenarios**: Define negotiation contexts and constraints
- **Data Export**: Export results in CSV and JSON formats

### 📊 Advanced Analytics
- **Conversation Analysis**: Detailed message-by-message breakdowns
- **Outcome Tracking**: Success rates, agreement prices, strategies
- **Performance Metrics**: Response times, negotiation lengths
- **Statistical Analysis**: Comprehensive data analysis tools

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- API keys for at least one LLM provider

### Quick Start

1. **Clone the Repository**
   ```bash
   git clone https://github.com/ayushi-uwc/epistemic_grounding_experiment.git
   cd epistemic_grounding_experiment
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API Keys**
   ```bash
   cp env.example .env
   # Edit .env with your API keys
   ```

4. **Run the Platform**
   ```bash
   python main.py
   ```

5. **Access the Interface**
   Open your browser to `http://localhost:5001`

## 🔑 API Key Configuration

### Environment Variables
Create a `.env` file with your API keys:

```bash
# Required: At least one API key
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
XAI_API_KEY=your_xai_api_key_here
GROQ_API_KEY=your_groq_api_key_here

# Optional: Flask configuration
FLASK_ENV=development
FLASK_DEBUG=True
FLASK_PORT=5001
```

### Frontend Configuration
Alternatively, configure API keys through the web interface:
1. Open the platform in your browser
2. Click "⚙️ API Keys" in the sidebar
3. Enter your API keys and test connections
4. Save configuration

## 📋 Usage Guide

### Single Negotiations

1. **Start a Negotiation**
   - Select buyer and seller AI models
   - Choose negotiation strategies
   - Click "🚀 Start Negotiation"

2. **Monitor Progress**
   - Watch real-time conversation
   - Track round-by-round exchanges
   - Monitor negotiation status

3. **Analyze Results**
   - View final outcomes
   - Export conversation data
   - Analyze negotiation patterns

### Batch Experiments

1. **Configure Batch Settings**
   - Define buyer/seller case combinations
   - Select LLM provider matrix
   - Set termination conditions

2. **Execute Batch**
   - Start automated experiments
   - Monitor progress dashboard
   - Track completion status

3. **Export Results**
   - Download comprehensive CSV data
   - Access detailed JSON logs
   - Generate analysis reports

## 🏗️ Project Structure

```
epistemic_grounding_experiment/
├── main.py                 # Main Flask application
├── config.py              # Configuration settings
├── requirements.txt       # Python dependencies
├── env.example            # Environment template
├── templates/
│   └── index.html         # Web interface
├── results/               # Experiment results (auto-generated)
├── docs/                  # Additional documentation
└── tests/                 # Test suite (future)
```

## 🔬 Research Methodology

### Negotiation Scenarios
The platform supports various negotiation contexts:
- **Car Sales**: Vehicle price negotiations with market dynamics
- **Real Estate**: Property transactions with multiple factors
- **Business Deals**: Contract negotiations with complex terms
- **Custom Scenarios**: User-defined negotiation contexts

### Strategy Variations
Different negotiation strategies can be tested:
- **Constrained**: Agents with strict budget/profit constraints
- **Unbounded**: Flexible negotiation parameters  
- **Symmetric**: Equal information between agents

### Data Collection
Comprehensive data is collected including:
- Complete conversation transcripts
- Offer progression and price movements
- Negotiation outcomes and success rates
- Timing and response patterns
- Strategy effectiveness analysis

## 📊 Data Analysis

The platform provides multiple data export formats:

### CSV Export
- **Summary Data**: Key metrics and outcomes
- **Detailed Data**: Complete conversation and analysis
- **Batch Results**: Aggregated experiment data

### JSON Export
- **Raw Conversations**: Complete message logs
- **Structured Analysis**: Parsed negotiation data
- **Metadata**: Experiment configuration and settings

## 🤝 Contributing

We welcome contributions from researchers, developers, and AI enthusiasts! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Areas for Contribution
- **New LLM Providers**: Add support for additional AI models
- **Analysis Tools**: Enhance data analysis capabilities
- **Negotiation Scenarios**: Create new experimental contexts
- **UI Improvements**: Enhance the web interface
- **Documentation**: Improve guides and examples



## ⚠️ Ethical Considerations

This platform is designed for research purposes. Please consider:
- **Fair Use**: Respect API provider terms of service
- **Data Privacy**: Handle conversation data responsibly
- **Research Ethics**: Follow institutional guidelines
- **Bias Awareness**: Consider AI model biases in results

## 🔧 Configuration Options

### LLM Provider Settings
```python
# config.py - Customize model parameters
LLM_PROVIDERS = {
    'openai': {
        'default_model': 'gpt-4o',
        'max_tokens': 2000,
        'temperature': 0.8
    },
    # ... other providers
}
```

### Negotiation Parameters
```python
# config.py - Adjust experiment settings
NEGOTIATION_CONFIG = {
    'max_turns': 40,
    'turn_delay': 3,
    'max_retries': 3
}
```

## 🐛 Troubleshooting

### Common Issues
1. **API Key Errors**: Verify keys are correctly configured
2. **Rate Limits**: Implement delays between requests
3. **Connection Issues**: Check internet connectivity
4. **Model Availability**: Ensure selected models are accessible

### Getting Help
- Open a [GitHub Issue](https://github.com/ayushi-uwc/epistemic_grounding_experiment/issues)
- Check [Discussions](https://github.com/ayushi-uwc/epistemic_grounding_experiment/discussions)
- Review [Documentation](docs/)

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Happy Experimenting!** 🔬✨

For questions, issues, or collaboration opportunities, please reach out through GitHub Issues or Discussions. 
