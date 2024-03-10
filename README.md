# iChat

# Conversational AI with Preprocessing and Gradio Interface

This project showcases a complete workflow for building a conversational AI model with text data preprocessing and a Gradio-based interface for interaction. It involves preprocessing text data using the Wikipedia dataset, ranking documents with BM25, and handling conversations through a chatbot interface powered by Langchain and Google Generative AI.

## Overview

The project is split into two main scripts:

- `preprocess.py`: Prepares the dataset for the conversational model. It loads, preprocesses, and ranks documents from the Wikipedia dataset.
- `interface_alpha.py`: Sets up a Gradio interface for real-time interaction with the chatbot, utilizing the preprocessed data and conversation history for context-aware responses.

## Dependencies

Ensure you have the following dependencies installed:

- torch
- torchtext
- langchain
- datasets (Hugging Face)
- rank_bm25
- tqdm
- numpy
- gradio
- langchain_google_genai

You can install these dependencies via pip:

```bash
pip install torch torchtext langchain datasets rank_bm25 tqdm numpy gradio langchain_google_genai
```

## Setup

To get started with this project, follow these setup instructions:

### 1. Google API Key

The application requires a Google API key to access Google's Generative AI models and embeddings. Make sure you have this key available before proceeding.

- Locate or create your Google API key.
- Save your key into a text file named `google_api.txt`.
- Place `google_api.txt` in the `PATH_DATA` directory specified in both scripts. The default path is mentioned as `/Users/deniskim/Library/CloudStorage/SynologyDrive-M1/문서/연구/DIAL/code/home/tako/minseok/dataset/`, but you should update this path to where you intend to store your dataset and API key.

### 2. Dataset Preparation

The `preprocess.py` script is designed to download and preprocess the Wikipedia dataset automatically if the preprocessed file `corpus_300k.txt` is not found. No additional manual dataset preparation is required.


## Usage

To use this project, you can execute the scripts either by running the provided `run.sh` shell script or manually executing each Python script. Here's how:

### Running with `run.sh`

The `run.sh` script simplifies the execution process by running the necessary commands sequentially. To use it, open a terminal in the project's root directory and run:

```bash
bash run.sh
```


This command executes `preprocess.py` followed by `interface_alpha.py`, automating the workflow from data preprocessing to interaction with the Gradio interface.

### Manual Execution

If you prefer to have more control over the process or wish to run the scripts independently, follow these steps:

1. **Preprocess the Data**

   Begin by executing the `preprocess.py` script to prepare your dataset:

   ```bash
   python preprocess.py
   ```


This will download and preprocess the dataset if necessary, setting the stage for the conversational AI to function properly. The preprocessing includes tasks such as dataset trimming, randomization, and cleaning, ensuring the data is in an optimal format for the model to generate meaningful responses.

2. **Launch the Gradio Interface**

After preparing the data, you can engage with the conversational AI through a user-friendly Gradio interface by running `interface_alpha.py`:

```bash
python interface_alpha.py
```

This command initiates a web interface, enabling users to input queries and receive responses from the AI model. The interface provides a convenient way to interact with the chatbot, facilitating real-time conversations that leverage the preprocessed data and stored conversation history for contextually aware interactions.


## Customization

To adapt the project to your specific needs or to experiment with its capabilities, consider the following customization options:

- **Directory Paths**: Adjust the `PATH` and `PATH_DATA` variables in both scripts to match your local file storage paths. This ensures the scripts can access necessary resources like the dataset and the Google API key file without any issues.
- **Dataset Query**: The hardcoded query in `preprocess.py` can be changed to explore how different queries affect the retrieval and ranking of documents. This can be particularly useful for tailoring the dataset preprocessing to better suit the conversational context you wish to explore with the AI.
- **Interface and Prompts**: In `interface_alpha.py`, the conversation prompts and the Gradio interface settings can be fine-tuned to modify the chatbot's interaction style and the user interface's appearance. Adjusting these settings allows for a more personalized or branded interaction experience, which can be beneficial for engaging users or for specific use cases.

### Implementing Customizations

To implement these customizations, follow these general steps:

1. **Edit the Scripts**: Open the `preprocess.py` and `interface_alpha.py` files in your preferred code editor.
2. **Modify Variables and Settings**: Locate the variables and settings you wish to change. For directory paths, look for the `PATH` and `PATH_DATA` variables. For changing the dataset query, find the query string within the `preprocess.py` script. To adjust the interface and prompts, navigate to the relevant sections in `interface_alpha.py`.
3. **Save Changes**: After making your adjustments, save the files.
4. **Test the Modifications**: Run the scripts again to ensure your changes have been successfully implemented and are working as expected. This may involve running the preprocessing step again or simply restarting the Gradio interface to test new interaction styles or prompts.

By customizing these aspects of the project, you can create a more tailored conversational AI experience that better fits your objectives or improves user engagement.

## Contribution

We welcome contributions from the community! Whether it's adding new features, fixing bugs, or improving documentation, your input is valuable. Here's how you can contribute:

1. **Fork the Repository**: Visit the project's GitHub page and click the 'Fork' button to create your own copy of the repository. This allows you to make changes without affecting the original project.
2. **Make Your Changes**: In your forked version of the project, implement your improvements or fixes. This could involve adding new functionality, correcting issues, or enhancing the documentation.
3. **Test Your Changes**: Ensure that your modifications work as intended and do not introduce new issues. This step is crucial for maintaining the quality of the project.
4. **Submit a Pull Request**: Once you're satisfied with your changes, submit a pull request to the original repository. Include a clear description of what you've done and why it adds value to the project.
5. **Participate in the Review Process**: Be open to feedback from the project maintainers and other contributors. This collaborative process helps ensure that contributions are beneficial and fit well with the project goals.

For any questions or to discuss your ideas before making changes, feel free to open an issue in the GitHub repository. Your contributions help make the project better for everyone!
