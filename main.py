import os
import time
import json
import random
import string
import requests
import argparse
import shutil
from collections import Counter
from typing import Dict, Any, List
import pymorphy3
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud, STOPWORDS

sw = set()
table = str.maketrans(dict.fromkeys(string.punctuation))
morph = pymorphy3.MorphAnalyzer()


def load_data_from_mm(host: str, channel_id: str, token: str, page_count: int, per_page: int) -> [dict]:
    """
    Load data from Mattermost and create a JSON file with the threads.
    """
    headers = {
        "Authorization": f"Bearer {token}",
    }

    pages = []
    for page in range(page_count):
        url = f"https://{host}/api/v4/channels/{channel_id}/posts"
        params = {
            "page": page,
            "per_page": per_page
        }
        response = requests.get(url, headers=headers, params=params)

        if response.status_code == 200:
            data = response.json()
            pages.append(data)
        else:
            raise Exception(f"Error: {response.status_code} - {response.text}")

        print(f"Page {page} loaded")

        time.sleep(0.1)

    with open(f'./data/pages.json', 'w+', encoding='utf-8') as f:
        json.dump(pages, f, ensure_ascii=False, indent=4)

    print("All pages loaded")

    return pages


def get_normalized_messages(pages: List[dict], sw: dict) -> List[str]:
    """
    Get messages from the pages.
    """
    messages = []
    for page in pages:
        for _, post in page['posts'].items():
            messages.append(normalize(post['message'], sw))

    with open(f'./data/messages.json', 'w+', encoding='utf-8') as f:
        json.dump(messages, f, ensure_ascii=False, indent=4)

    print("All messages normalized")

    return messages


def unpack_tokens(tokens: List[List[str]]) -> List[str]:
    return [token for sublist in tokens for token in sublist]


def normalize(msg: str, sw: set) -> List[str]:
    """
    Normalize the given message by removing punctuation, tokenizing words, lemmatizing words, removing stopwords and digits.

    Args:
        msg (str): The message to be normalized.

    Returns:
        List[str]: The list of filtered tokens after normalization.
    """
    # Remove punctuation
    text_without_punct = msg.translate(
        str.maketrans('', '', string.punctuation))

    # Tokenize words
    tokens = word_tokenize(text_without_punct)

    # Lemmatize words
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(
        token.lower()) for token in tokens]

    # Remove stopwords and digits
    filtered_tokens = [token for token in lemmatized_tokens if token.lower(
    ) not in sw and not token.isdigit()]

    return filtered_tokens


def generate_cloud(data: List[str], out_file: str, include_numbers: bool, width: int, height: int) -> None:
    """
    Generate a word cloud image based on the frequency of words in the input data.

    Args:
        data (List[str]): List of strings representing the input data.
        out_file (str): Path to save the generated word cloud image.
    """
    freq = Counter(data)

    wc = WordCloud(width=width, height=height, max_words=200, normalize_plurals=True,
                   include_numbers=include_numbers, margin=10, random_state=1).generate_from_frequencies(freq)

    fig, ax = plt.subplots()
    ax.imshow(wc.recolor(random_state=int(random.random() * 256)),
              interpolation="bilinear")
    ax.set_title("Chat Word Cloud")
    ax.axis("off")

    plt.savefig(out_file)
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Generate word cloud from MM channels')
    parser.add_argument('-sw', '--stop_words', type=str,
                        help='Path to custom csv file with stop words')
    parser.add_argument('-c', '--channel_id', type=str,
                        help='MM channel ID', required=True)
    parser.add_argument('-tk', '--token', type=str,
                        help='MM token', required=True)
    parser.add_argument('-host', '--host', type=str,
                        help='MM host', required=True)
    parser.add_argument('-p', '--pages', type=int,
                        help='Number of pages to load from the channel', default=100)
    parser.add_argument('-pp', '--per_page', type=int,
                        help='Number of messages per page', default=100)
    parser.add_argument('-t', '--timeout', type=str,
                        help='Delay between MM requests (increase if downloading a lot)', default=0.5)
    parser.add_argument('-o', '--out_file', type=str,
                        help='Output file path', default='./data/out.png')
    parser.add_argument('-i', '--include_numbers', action='store_true',
                        help='Include word count in the word cloud', default=False)
    parser.add_argument('-f', '--force', action='store_true',
                        help='Overwrite downloaded data', default=False)
    parser.add_argument('-width', '--width', type=int,
                        help='Width of the word cloud image', default=2560)
    parser.add_argument('-height', '--height', type=int,
                        help='Height of the word cloud image', default=1440)

    args = parser.parse_args()

    if not os.path.exists('./data'):
        os.makedirs('./data')

    sw = set(stopwords.words('russian')) | set(stopwords.words(
        'english')) | set(STOPWORDS)

    if args.stop_words is not None:
        with open(args.stop_words, 'r', encoding='utf-8') as f:
            stop_words = f.read().splitlines()
        sw = sw | set(stop_words)

    if args.force:
        if os.path.exists('./data'):
            shutil.rmtree('./data')

    if os.path.exists('./data/pages.json'):
        with open('./data/pages.json', 'r') as f:
            mm_msgs = json.load(f)
    else:
        mm_msgs = load_data_from_mm(
            args.host, args.channel_id, args.token, args.pages, args.per_page)

    if os.path.exists('./data/messages.json'):
        with open('./data/messages.json', 'r') as f:
            normalized_msg = json.load(f)
    else:
        normalized_msg = get_normalized_messages(mm_msgs, sw)

    tokens = unpack_tokens(normalized_msg)
    generate_cloud(tokens, args.out_file, args.include_numbers,
                   args.width, args.height)

    print("Word cloud generated")


if __name__ == "__main__":
    main()
