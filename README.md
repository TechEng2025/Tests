# Tests
Here's the complete end-to-end implementation with all components integrated:

### Backend (Spring Boot)

1. **pom.xml**:
```xml
<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0">
    <modelVersion>4.0.0</modelVersion>
    <parent>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-parent</artifactId>
        <version>3.2.0</version>
    </parent>
    
    <groupId>com.sentiment</groupId>
    <artifactId>backend</artifactId>
    <version>1.0.0</version>
    <name>Sentiment Analysis Backend</name>

    <properties>
        <java.version>21</java.version>
        <onnxruntime.version>1.16.0</onnxruntime.version>
    </properties>

    <dependencies>
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-webflux</artifactId>
        </dependency>
        <dependency>
            <groupId>com.microsoft.onnxruntime</groupId>
            <artifactId>onnxruntime</artifactId>
            <version>${onnxruntime.version}</version>
        </dependency>
        <dependency>
            <groupId>org.apache.commons</groupId>
            <artifactId>commons-text</artifactId>
            <version>1.10.0</version>
        </dependency>
    </dependencies>

    <build>
        <plugins>
            <plugin>
                <groupId>org.springframework.boot</groupId>
                <artifactId>spring-boot-maven-plugin</artifactId>
                <configuration>
                    <jvmArguments>
                        --enable-preview -XX:+UseZGC
                    </jvmArguments>
                </configuration>
            </plugin>
            <plugin>
                <groupId>org.apache.maven.plugins</groupId>
                <artifactId>maven-compiler-plugin</artifactId>
                <configuration>
                    <compilerArgs>--enable-preview</compilerArgs>
                </configuration>
            </plugin>
        </plugins>
    </build>
</project>
```

2. **Application.java**:
```java
package com.sentiment;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class Application {
    public static void main(String[] args) {
        SpringApplication.run(Application.class, args);
    }
}
```

3. **ONNXModel.java** (Model Inference):
```java
package com.sentiment.onnx;

import ai.onnxruntime.*;
import org.springframework.core.io.ClassPathResource;
import org.springframework.stereotype.Component;
import javax.annotation.PostConstruct;
import java.util.*;

@Component
public class ONNXModel {
    private OrtEnvironment env;
    private OrtSession session;
    private final Map<String, Integer> vocab = new HashMap<>();
    private final int MAX_LEN = 128;

    @PostConstruct
    public void init() throws Exception {
        this.env = OrtEnvironment.getEnvironment();
        ClassPathResource modelResource = new ClassPathResource("models/sentiment_model.onnx");
        this.session = env.createSession(modelResource.getInputStream(), new OrtSession.SessionOptions());
        
        // Load vocabulary (in real implementation, load from file)
        List<String> words = List.of("[PAD]", "[UNK]", "[CLS]", "[SEP]", "good", "bad", "hotel", ...);
        for (int i = 0; i < words.size(); i++) {
            vocab.put(words.get(i), i);
        }
    }

    public float predict(String text) throws OrtException {
        // Tokenize and convert to input tensors
        List<Integer> tokens = tokenize(text);
        long[] inputIds = new long[MAX_LEN];
        long[] attentionMask = new long[MAX_LEN];
        
        // [CLS] token
        inputIds[0] = vocab.getOrDefault("[CLS]", 1);
        attentionMask[0] = 1;
        
        // Add tokens
        for (int i = 0; i < Math.min(tokens.size(), MAX_LEN-1); i++) {
            inputIds[i+1] = tokens.get(i);
            attentionMask[i+1] = 1;
        }
        
        // Create ONNX inputs
        OnnxTensor idsTensor = OnnxTensor.createTensor(env, new long[][]{inputIds});
        OnnxTensor maskTensor = OnnxTensor.createTensor(env, new long[][]{attentionMask});
        
        Map<String, OnnxTensor> inputs = Map.of(
            "input_ids", idsTensor,
            "attention_mask", maskTensor
        );

        try (OrtSession.Result results = session.run(inputs)) {
            float[][] output = (float[][]) results.get(0).getValue();
            return output[0][1]; // Positive sentiment probability
        }
    }

    private List<Integer> tokenize(String text) {
        // Simplified tokenization (use WordPiece in real implementation)
        return Arrays.stream(text.split("\\s+"))
            .map(word -> vocab.getOrDefault(word.toLowerCase(), 1)) // 1 = [UNK]
            .limit(MAX_LEN - 2)
            .toList();
    }
}
```

4. **SentimentController.java**:
```java
package com.sentiment.controller;

import com.sentiment.onnx.ONNXModel;
import com.sentiment.model.PredictionResponse;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;
import reactor.core.publisher.Mono;

@RestController
@RequestMapping("/api/sentiment")
public class SentimentController {

    @Autowired
    private ONNXModel model;

    @PostMapping("/predict")
    public Mono<PredictionResponse> predict(@RequestBody String review) {
        try {
            float score = model.predict(review);
            return Mono.just(new PredictionResponse(score));
        } catch (Exception e) {
            return Mono.error(e);
        }
    }
}
```

5. **PredictionResponse.java**:
```java
package com.sentiment.model;

public record PredictionResponse(float score) {}
```

6. **AsyncConfig.java** (Virtual Threads):
```java
package com.sentiment.config;

import org.springframework.context.annotation.Configuration;
import org.springframework.scheduling.annotation.AsyncConfigurer;
import org.springframework.scheduling.annotation.EnableAsync;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

@Configuration
@EnableAsync
public class AsyncConfig implements AsyncConfigurer {
    @Override
    public Executor getAsyncExecutor() {
        return Executors.newVirtualThreadPerTaskExecutor();
    }
}
```

7. **application.properties**:
```properties
server.port=8080
spring.main.web-application-type=reactive
```

### Frontend (Next.js)

1. **package.json**:
```json
{
  "name": "frontend",
  "version": "1.0.0",
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start"
  },
  "dependencies": {
    "chart.js": "^4.4.0",
    "next": "^14.1.0",
    "react": "^18.2.0",
    "react-chartjs-2": "^5.2.0",
    "react-dom": "^18.2.0"
  }
}
```

2. **pages/index.js**:
```jsx
import { useState } from 'react';
import RealTimeSentiment from '../components/RealTimeSentiment';
import Dashboard from '../components/Dashboard';

export default function Home() {
  const [sentimentData, setSentimentData] = useState([]);

  const handleNewPrediction = (score) => {
    setSentimentData(prev => [
      ...prev.slice(-9), 
      { score, timestamp: new Date
