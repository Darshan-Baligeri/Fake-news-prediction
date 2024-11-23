# Fake News Detection Flask Application

This is a Flask web application for detecting fake news using a pre-trained machine learning model. The application provides a web interface for users to input news content and receive predictions on whether the news is real or fake.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Routes](#routes)
- [Model and Data Preparation](#model-and-data-preparation)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. **Clone the repository**:
    ```sh
    git clone https://github.com/yourusername/fake-news-detection.git
    cd fake-news-detection
    ```

2. **Create a virtual environment** (optional but recommended):
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required packages**:
    ```sh
    pip install -r requirements.txt
    ```

4. **Download NLTK stopwords**:
    ```sh
    python -m nltk.downloader stopwords
    ```

5. **Ensure you have the dataset and model files**:
    - Place the `train.csv` dataset file in the appropriate directory (`D:/ADS mini project/fake-news/`).
    - Place the `model.keras` file in the root directory of the project.

## Usage

1. **Run the application**:
    ```sh
    python app.py
    ```

2. **Open your browser** and navigate to `http://127.0.0.1:5000` to access the web interface.

## Project Structure
fake-news-detection/
│
├── static/
│ └── (static files such as CSS, JavaScript, images)
│
├── templates/
│ ├── home.html
│ ├── about.html
│ ├── news.html
│ └── contact.html
│
├── model.keras
├── train.csv
├── app.py
└── requirements.txt


## Routes

- **`/`**: Home page.
- **`/about`**: About page.
- **`/news`**: News page.
- **`/contact`**: Contact page.
- **`/predict`**: API endpoint to predict if the news is real or fake. Expects a POST request with JSON data containing the `content` field.

## Model and Data Preparation

- **Dataset**: The application uses a dataset containing news articles with labels indicating if they are real or fake.
- **Preprocessing**: The news content is preprocessed using stemming and stopword removal.
- **Vectorization**: The content is vectorized using TF-IDF.
- **Model**: A pre-trained TensorFlow model (`model.keras`) is used to predict if the news is real or fake.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any changes or enhancements.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.