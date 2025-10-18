"""
Message preprocessing utilities for AbstractCore.

This module provides utilities for parsing and preprocessing user messages,
particularly for extracting file references using @filename syntax.
Used across all AbstractCore applications for consistent behavior.
"""

import re
import os
from typing import Tuple, List, Optional


class MessagePreprocessor:
    """
    Message preprocessing utilities for extracting file references and cleaning input.

    Supports @filename syntax for attaching files to messages across all AbstractCore apps.
    """

    # Pattern to match @filename (supports various file extensions)
    FILE_PATTERN = r'@([^\s@]+\.[\w]+)'

    @staticmethod
    def parse_file_attachments(user_input: str,
                             validate_existence: bool = True,
                             verbose: bool = False) -> Tuple[str, List[str]]:
        """
        Parse @filename references from user input and return cleaned text + file list.

        Args:
            user_input: The user's message that may contain @filename references
            validate_existence: Whether to check if files actually exist (default: True)
            verbose: Whether to log file processing details (default: False)

        Returns:
            Tuple of (clean_input_text, list_of_valid_file_paths)

        Examples:
            >>> clean_text, files = MessagePreprocessor.parse_file_attachments(
            ...     "Analyze this image @screenshot.png and @data.csv"
            ... )
            >>> print(clean_text)
            "Analyze this image  and"
            >>> print(files)
            ["screenshot.png", "data.csv"]
        """
        # Find all @filename references
        matches = re.findall(MessagePreprocessor.FILE_PATTERN, user_input)

        if not matches:
            return user_input, []

        valid_files = []
        invalid_files = []

        for filename in matches:
            if not validate_existence or os.path.exists(filename):
                valid_files.append(filename)
                if verbose:
                    size_kb = os.path.getsize(filename) / 1024 if validate_existence else 0
                    print(f"📎 Found file: {filename} ({size_kb:.1f}KB)")
            else:
                invalid_files.append(filename)

        # Show warnings for missing files if verbose mode
        if verbose and invalid_files:
            print(f"⚠️  Files not found: {', '.join(invalid_files)}")

        # Remove @filename references from the input text
        clean_input = re.sub(MessagePreprocessor.FILE_PATTERN, '', user_input)

        # Clean up extra whitespace
        clean_input = re.sub(r'\s+', ' ', clean_input).strip()

        return clean_input, valid_files

    @staticmethod
    def has_file_attachments(user_input: str) -> bool:
        """
        Check if the user input contains any @filename references.

        Args:
            user_input: The message to check

        Returns:
            True if @filename patterns are found, False otherwise
        """
        return bool(re.search(MessagePreprocessor.FILE_PATTERN, user_input))

    @staticmethod
    def get_file_count(user_input: str) -> int:
        """
        Count the number of @filename references in the input.

        Args:
            user_input: The message to analyze

        Returns:
            Number of @filename patterns found
        """
        return len(re.findall(MessagePreprocessor.FILE_PATTERN, user_input))

    @staticmethod
    def extract_file_paths(user_input: str) -> List[str]:
        """
        Extract just the file paths from @filename references without validation.

        Args:
            user_input: The message containing @filename references

        Returns:
            List of file paths (may include non-existent files)
        """
        return re.findall(MessagePreprocessor.FILE_PATTERN, user_input)

    @staticmethod
    def process_message_with_media(user_input: str,
                                 default_prompt: Optional[str] = None,
                                 validate_files: bool = True,
                                 verbose: bool = False) -> Tuple[str, List[str]]:
        """
        Process a message with @filename attachments, providing a default prompt if needed.

        This is the main entry point for applications that want full message preprocessing.

        Args:
            user_input: The user's message
            default_prompt: Default text to use if only files are specified (e.g., "Analyze the attached files")
            validate_files: Whether to validate file existence
            verbose: Whether to show processing details

        Returns:
            Tuple of (processed_prompt, media_file_list)
        """
        clean_input, media_files = MessagePreprocessor.parse_file_attachments(
            user_input,
            validate_existence=validate_files,
            verbose=verbose
        )

        # If no text remains after removing file references, use default prompt
        if not clean_input and media_files and default_prompt:
            clean_input = default_prompt

        return clean_input, media_files


# Convenience functions for common use cases
def parse_files(user_input: str, verbose: bool = False) -> Tuple[str, List[str]]:
    """
    Convenience function for basic file parsing.

    Args:
        user_input: Message with @filename references
        verbose: Show processing details

    Returns:
        Tuple of (clean_text, file_list)
    """
    return MessagePreprocessor.parse_file_attachments(user_input, verbose=verbose)


def has_files(user_input: str) -> bool:
    """
    Convenience function to check if message has file attachments.

    Args:
        user_input: Message to check

    Returns:
        True if @filename patterns found
    """
    return MessagePreprocessor.has_file_attachments(user_input)


# Export main classes and functions
__all__ = [
    'MessagePreprocessor',
    'parse_files',
    'has_files'
]