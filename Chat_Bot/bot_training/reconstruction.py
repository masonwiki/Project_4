import json

# Read the existing JSON file
with open('intents2.json', 'r') as input_file:
    input_data = json.load(input_file)

# Assuming your input data is a dictionary with an "intents" key containing a list of dictionaries
if "intents" in input_data and isinstance(input_data["intents"], list):
    intents_data = input_data["intents"]

    # Create a dictionary to store the transformed data
    transformed_data = []

    # Iterate through the "intents" data and structure it as desired
    for item in intents_data:
        if "tag" in item and "patterns" in item and "responses" in item:
            transformed_item = {
                "tag": item["tag"],
                "patterns": item["patterns"],
                "responses": item["responses"]
            }
            transformed_data.append(transformed_item)

    # Create a new dictionary with the "intents" key and the transformed data
    output_data = {"intents": transformed_data}

    # Write the transformed data to a new JSON file with proper indentation
    with open('intents3.json', 'w') as output_file:
        json.dump(output_data, output_file, indent=2)  # Adjust the indentation level as needed

    print("Transformation completed. JSON data saved to 'your_output_file.json'")
else:
    print("Invalid input JSON format. 'intents' key not found or not in the expected format.")