# Tests
Here's the complete **end-to-end implementation** with proper `fetch` usage for all API calls, optimized for the hotel sentiment analysis system:

### **1. Backend (Spring Boot WebFlux)**

#### **SentimentController.java** (Updated with CORS)
```java
@RestController
@RequestMapping("/api")
@CrossOrigin(origins = "*") // Enable CORS for frontend
public class SentimentController {
    private final OnnxInferenceService inferenceService;
    
    @PostMapping(value = "/analyze", 
                consumes = MediaType.TEXT_PLAIN_VALUE,
                produces = MediaType.APPLICATION_JSON_VALUE)
    public Mono<SentimentResult> analyze(@RequestBody String text) {
        return inferenceService.predict(text)
            .map(scores -> new SentimentResult(
                scores[0], // negative
                scores[1], // neutral
                scores[2], // positive
                getSentimentLabel(scores),
                (Arrays.stream(scores).max().getAsFloat() / Arrays.stream(scores).sum()) * 100
            ));
    }
    
    private String getSentimentLabel(float[] scores) {
        int maxIndex = IntStream.range(0, scores.length)
            .reduce((i,j) -> scores[i] > scores[j] ? i : j)
            .orElse(0);
        return switch(maxIndex) {
            case 0 -> "NEGATIVE";
            case 1 -> "NEUTRAL";
            case 2 -> "POSITIVE";
            default -> throw new IllegalStateException();
        };
    }
}
```

### **2. Frontend (Next.js)**

#### **pages/index.jsx** (Enhanced with Fetch)
```jsx
import { useState } from 'react'
import dynamic from 'next/dynamic'
const SentimentGlobe = dynamic(() => import('../components/SentimentGlobe'), { 
    ssr: false,
    loading: () => <div className="loading">Loading 3D viewer...</div>
})

export default function Home() {
    const [results, setResults] = useState(null)
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState(null)

    const analyzeReview = async (text) => {
        if (!text.trim()) return;
        
        setLoading(true)
        setError(null)
        
        try {
            const response = await fetch('http://localhost:8080/api/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'text/plain', // Match backend consumption
                },
                body: text
            })
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`)
            }
            
            const data = await response.json()
            setResults(data)
        } catch (err) {
            setError(err.message)
            console.error("Analysis failed:", err)
        } finally {
            setLoading(false)
        }
    }

    return (
        <div className="container">
            <div className="input-section">
                <h1>Hotel Review Sentiment Analysis</h1>
                <textarea
                    placeholder="Paste hotel review here..."
                    onChange={(e) => analyzeReview(e.target.value)}
                    disabled={loading}
                />
                {loading && <div className="spinner"></div>}
                {error && <div className="error">{error}</div>}
            </div>
            
            <div className="visualization">
                {results && (
                    <>
                        <SentimentGlobe data={results} />
                        <div className="results">
                            <h2>Analysis Results</h2>
                            <p>
                                <strong>Sentiment:</strong> 
                                <span className={`tag ${results.sentiment.toLowerCase()}`}>
                                    {results.sentiment}
                                </span>
                            </p>
                            <p><strong>Confidence:</strong> {results.confidence.toFixed(1)}%</p>
                            <div className="score-bars">
                                <div className="score negative" style={{ width: `${results.negativeScore * 100}%` }}>
                                    Negative: {(results.negativeScore * 100).toFixed(1)}%
                                </div>
                                <div className="score neutral" style={{ width: `${results.neutralScore * 100}%` }}>
                                    Neutral: {(results.neutralScore * 100).toFixed(1)}%
                                </div>
                                <div className="score positive" style={{ width: `${results.positiveScore * 100}%` }}>
                                    Positive: {(results.positiveScore * 100).toFixed(1)}%
                                </div>
                            </div>
                        </div>
                    </>
                )}
            </div>
        </div>
    )
}
```

#### **styles/globals.css**
```css
.container {
    display: flex;
    min-height: 100vh;
}

.input-section {
    width: 40%;
    padding: 2rem;
}

.visualization {
    width: 60%;
    position: relative;
}

textarea {
    width: 100%;
    height: 200px;
    padding: 1rem;
    font-size: 1rem;
    border: 2px solid #ddd;
    border-radius: 8px;
    resize: none;
}

.loading, .error {
    margin-top: 1rem;
    padding: 1rem;
    border-radius: 4px;
}

.error {
    background-color: #ffebee;
    color: #d32f2f;
}

.tag {
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    color: white;
    margin-left: 0.5rem;
}

.tag.negative { background-color: #ff5a5f; }
.tag.neutral { background-color: #ffd700; color: #333; }
.tag.positive { background-color: #00a699; }

.score-bars {
    display: flex;
    height: 30px;
    border-radius: 15px;
    overflow: hidden;
    margin: 1rem 0;
}

.score {
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-size: 0.8rem;
    font-weight: bold;
}

.negative { background-color: #ff5a5f; }
.neutral { background-color: #ffd700; color: #333; }
.positive { background-color: #00a699; }

@keyframes spin {
    to { transform: rotate(360deg); }
}

.spinner {
    display: inline-block;
    width: 20px;
    height: 20px;
    border: 3px solid rgba(0,0,0,0.1);
    border-radius: 50%;
    border-top-color: #00a699;
    animation: spin 1s ease-in-out infinite;
    margin-left: 1rem;
}
```

### **3. Machine Learning (Python)**

#### **train.py** (Add Confidence Calibration)
```python
# Add to training script
class ConfidenceCalibrator:
    def __init__(self, model):
        self.model = model
        self.temperature = tf.Variable(1.0, trainable=True)
    
    def calibrate(self, val_data, epochs=5):
        optimizer = tf.keras.optimizers.Adam(0.01)
        for _ in range(epochs):
            for x, y in val_data:
                with tf.GradientTape() as tape:
                    logits = self.model(x) / self.temperature
                    loss = tf.keras.losses.categorical_crossentropy(y, logits)
                grads = tape.gradient(loss, [self.temperature])
                optimizer.apply_gradients(zip(grads, [self.temperature]))
    
    def predict(self, x):
        logits = self.model(x) / self.temperature
        return tf.nn.softmax(logits)

# Usage after training
calibrator = ConfidenceCalibrator(model)
calibrator.calibrate(validation_data)
```

### **4. System Startup**

#### **Backend**
```bash
cd backend
mvn clean package
java -XX:+UseZGC -Xmx4G -jar target/backend.jar
```

#### **Frontend**
```bash
cd frontend
npm install
npm run dev
```

### **Key Features**:

1. **Proper Fetch Usage**:
   - Correct `Content-Type: text/plain` header
   - Error handling and loading states
   - CORS configuration on backend

2. **Enhanced UI**:
   - Real-time sentiment visualization
   - Confidence score display
   - Responsive design

3. **Optimized Backend**:
   - ZGC memory management
   - Virtual thread support
   - ONNX model inference

4. **Complete Workflow**:
   - Text input → API call → 3D visualization
   - Score breakdown display
   - Loading/error states

This implementation provides a **production-ready** system with proper API communication between frontend and backend, including all necessary optimizations and user experience considerations.
