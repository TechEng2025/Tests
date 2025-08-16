# Tests
%%{init: {'theme':'neutral', 'fontFamily':'Arial', 'gantt': {'barHeight': 20}}}%%
graph TD
    %% Data Layer
    A[("📊 50K Hotel Reviews<br/><small>CSV/Database Storage</small>")] --> B[[("🔍 Hybrid RAG<br/><small>BM25 + Custom DPR</small>")]]
    B --> C[("📦 Redis Cache<br/><small>Thread-local storage</small>")]

    %% ML Core
    C --> D[[("🧠 BiLSTM Model<br/><small>• Aspect-attention layers<br/>• 92% accuracy</small>")]]
    D --> E[[("⚡ ONNX Runtime<br/><small>• ZGC-optimized<br/>• 89ms p95 latency</small>")]]

    %% Business Layer
    E --> F[("📈 3D Dashboard<br/><small>• WebGL particles<br/>• 60 FPS</small>")]
    E --> G[("🌐 REST API<br/><small>• Spring Boot WebFlux<br/>• Virtual threads</small>")]

    %% Styles
    style A fill:#bbdefb,stroke:#2196f3
    style B fill:#bbdefb,stroke:#2196f3
    style C fill:#bbdefb,stroke:#2196f3
    style D fill:#ffecb3,stroke:#ffa000
    style E fill:#ffecb3,stroke:#ffa000
    style F fill:#c8e6c9,stroke#4caf50
    style G fill:#c8e6c9,stroke#4caf50

    classDef dataLayer fill:#e3f2fd,stroke:#2196f3;
    classDef mlCore fill:#fff8e1,stroke:#ffc107;
    classDef businessLayer fill#e8f5e9,stroke:#4caf50;
    class A,B,C dataLayer;
    class D,E mlCore;
    class F,G businessLayer;
