# Plant Disease Detection API

## Setup

> **Note** This project is tested on Python 3.11, but it should work on Python 3.10 and above.

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the server

```bash
python app.py
```

## Routes

### `GET /`
A get request to `/` will return `{ "message": "Ping" }`.
```bash
curl -X GET http://localhost:8888/
```

### `POST /`
A post request to `/` will return `{ "prediction": <Predicted_Class_Name> }`.
Or in case of an error, `{ "error": <Error_Message> }`. 
```bash
curl -X POST -F image=@<path_to_image> http://localhost:8888/
```

## File Structure

```
.
├── app.py - The main file that contains the API routes ( Flask server ).
├── constants.py - Contains the class names and the model path.
├── model.py - Contains the functions to load the model and predict the class.
├── model_scripted_cpu.pt - The model file saved as Pytorch Scripted Model.
├── requirements.txt - Contains the dependencies.
├── README.md - This file.
├── test.jpg - A test image.
```

> **Note** you can run `python model.py` to test the model with the test image. If you want to test with 
> another image replace the `test.jpg` with your image file.
