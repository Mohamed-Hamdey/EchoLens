# AI Subtitles for Deaf People

## Overview
This project aims to build an affordable AI-based system that converts speech into real-time subtitles, displaying them on a pair of glasses for people who are deaf or hard of hearing.

## Features
✅ Real-time speech-to-text conversion  
✅ Offline and online speech recognition  
✅ Display text output on a small screen attached to glasses  
✅ Open-source and budget-friendly implementation  

## Project Goals
1. **Develop a software system** that captures audio, converts it to text, and displays it.
2. **Find low-cost hardware solutions** to make it accessible for everyone.
3. **Open-source the project** to allow others to contribute and improve it.

## Technology Stack
- **Programming Language:** Python
- **Speech Recognition:** Google API / Vosk (offline)
- **Display Handling:** Pygame (for simulation), OLED Screen (for final hardware)
- **Hardware (Planned):** Raspberry Pi, ESP32, or Arduino with a small display

## Project Structure
```
AI-Subtitles-For-Deaf/
│── hardware/               # Documentation for hardware setup
│── software/               # Source code for speech-to-text processing
│   ├── models/             # AI models (Vosk, etc.)
│   ├── scripts/            # Python scripts
│   ├── display/            # Code for displaying subtitles
│   ├── README.md           # Explanation of software
│── docs/                   # Technical documentation
│── tests/                  # Testing scripts
│── LICENSE                 # Open-source license
│── README.md               # Project overview
```

## Getting Started
### 1. Clone the Repository
```
git clone https://github.com/YOUR_USERNAME/AI-Subtitles-For-Deaf.git
cd AI-Subtitles-For-Deaf
```
### 2. Install Dependencies
```
pip install -r requirements.txt
```

### 3. Run the Speech-to-Text System
```
python software/scripts/speech_to_text.py
```

## Contributing
1. Fork the repository
2. Create a new branch (`feature-xyz`)
3. Commit your changes
4. Push to GitHub and create a pull request

## License
This project is licensed under the MIT License.

