def read_prompt(file_path: str) -> str:
    try:
        with open(file_path, 'r') as file:
            prompt = file.read().strip()
        return prompt
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
