from langdetect import detect, DetectorFactory, LangDetectException
import re

DetectorFactory.seed = 0  # Ensures consistent results


def read_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def write_file(file_path, content):
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(content)


def remove_words_regex(input_string, list_of_strs):
    """
    Removes specified words from the content using regex.

    Args:
        input_string (str): Input text content.
        list_of_strs (list of str): Words to remove.

    Returns:
        str: Updated content with the specified words removed.
    """
    # Create a regex pattern dynamically from the word list
    the_re_pattern = r'\b(' + '|'.join(map(re.escape, list_of_strs)) + r')\b'
    modified_string = re.sub(the_re_pattern, '', input_string)
    return modified_string


def remove_french(the_text):
    """
    Removes paragraphs detected as French ('fr') from the input text, preserving the original document structure and whitespace.

    Args:
        the_text (str): The input text with paragraphs separated by newlines.

    Returns:
        str: The modified text with French paragraphs removed, while preserving original structure and whitespace.
    """
    paragraphs = the_text.split('\n')  # Assuming paragraphs are separated by newlines
    result_paragraphs = []

    for p in paragraphs:
        if not p:  # Keep empty paragraphs to preserve structure
            result_paragraphs.append(p)
            continue
        try:
            if detect(p) != 'fr':  # Detect language and exclude French
                result_paragraphs.append(p)
        except LangDetectException:
            # If language can't be detected, keep the paragraph as is
            result_paragraphs.append(p)

    return '\n'.join(result_paragraphs)


def split_into_subsections(input_text):
    """
    Splits a long string into subsections based on markers like [1], [2], [3],... [n].
    Includes any text before the first marker as its own subsection.

    Args:
        input_text (str): The input string with markers in the format [1], [2], etc.

    Returns:
        list of str: A list of strings, each representing a subsection of the text.
    """
    # Regex pattern to match markers like [1], [2], etc.
    pattern = r'\[\d+\]'

    # Find all matches for the pattern
    matches = list(re.finditer(pattern, input_text))

    # If no markers are found, return the entire text as a single subsection
    if not matches:
        return [input_text]

    subsections = []
    start = 0

    # Loop through the markers to extract sections
    for match in matches:
        end = match.start()
        # Append the text before the current marker
        if start != end:
            subsections.append(input_text[start:end].strip())
        # Start the next section with the current marker
        start = end

    # Append the last section after the final marker
    subsections.append(input_text[start:].strip())

    # check the len of each subsection, if >20, keep it
    list_of_strings_to_return = [section for section in subsections if len(section) >= 20]

    return list_of_strings_to_return

