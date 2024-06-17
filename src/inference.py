import torch

from src.data_utils import sentence_to_tensor


def perform_inference(model, sentence, word_to_index, tag_to_index, device, tensor_type):
    """
    Perform inference on the trained BoW model.

    Args:
        model (torch.nn.Module): The trained BoW model.
        sentence (str): The input sentence for inference.
        word_to_index (dict): A dictionary mapping words to their indices.
        tag_to_index (dict): A dictionary mapping tags to their indices.
        device (str): "cuda" or "cpu" based on availability.
        tensor_type (tensortype): type of torch.LongTensor
    Returns:
        str: The predicted class/tag for the input sentence.
    """
    # Preprocess the input sentence to match the model's input format
    sentence_tensor = sentence_to_tensor(sentence, word_to_index, tensor_type)

    # Move the input tensor to the same device as the model
    sentence_tensor = sentence_tensor.to(device)

    # Make sure the model is in evaluation mode and on the correct device
    model.eval()
    model.to(device)

    # Perform inference
    with torch.no_grad():
        output = model(sentence_tensor)

    # Move the output tensor to CPU if it's on CUDA
    if device == 'cuda':
        output = output.cpu()

    # Convert the model's output to a predicted class/tag
    predicted_class = torch.argmax(output).item()

    # Reverse lookup to get the tag corresponding to the predicted class
    for tag, index in tag_to_index.items():
        if index == predicted_class:
            return tag

    # Return an error message if the tag is not found
    return 'Tag not found'
