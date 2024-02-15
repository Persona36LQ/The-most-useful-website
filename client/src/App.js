import React, {useState} from 'react';
import axios from "axios";
import "./App.css"

const App = () => {
    const [uploadedImage, setUploadedImage] = useState(null);
    const [result, setResult] = useState(null);

    const handleImageUpload = (event) => {
        const file = event.target.files[0];
        setUploadedImage(file);
    };

    const handleImageSubmit = async () => {
        const formData = new FormData();
        formData.append('img', uploadedImage);

        try {
            const response = await axios.post("http://127.0.0.1:8000/api/image/", formData, {
                headers: {
                    'Content-Type': 'multipart/form-data'
                }
            });

            console.log(response.data)

            setResult(response.data); // Предположим, что сервер отправляет текстовый ответ
        } catch (error) {
            console.error('Ошибка при отправке изображения:', error);
        }
    };

    return (
        <div className="app">
            <h1 className="title">The most useful website.</h1>
            <br/>
            <h1 className="title"> Provide a photo of cat or dog and neural network will tell you what is on this picture</h1>
            <input type="file" accept="image/*" onChange={handleImageUpload} className="file-input"/>
            {uploadedImage && (
                <div className="preview-container">
                    <button onClick={handleImageSubmit} className="submit-button">Submit</button>
                </div>
            )}
            {result !== null && (
                <div className="result-container">
                    <h1 className="result-title">{result}</h1>
                </div>
            )}
        </div>
    );

};

export default App;
