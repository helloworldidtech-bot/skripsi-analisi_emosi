import streamlit as st
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import pandas as pd
import re
import io
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Emotion Classifier - Electronics Review",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .sub-header {
        font-size: 1.3rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    .warning-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .error-box {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .success-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .info-box {
        background-color: #d1ecf1;
        border-left: 5px solid #17a2b8;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem;
        font-size: 1.1rem;
        border-radius: 10px;
        font-weight: bold;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    .upload-section {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Electronics keywords filter
ELECTRONICS_KEYWORDS = {
    # Perangkat Elektronik
    'laptop', 'komputer', 'pc', 'notebook', 'macbook', 'smartphone', 'handphone', 'hp', 'iphone',
    'samsung', 'xiaomi', 'oppo', 'vivo', 'realme', 'asus', 'lenovo', 'acer', 'dell', 'apple',
    
    # Komponen & Aksesori
    'charger', 'kabel', 'adaptor', 'powerbank', 'baterai', 'battery', 'headset', 'earphone',
    'headphone', 'speaker', 'mouse', 'keyboard', 'monitor', 'layar', 'screen', 'display',
    'ram', 'hardisk', 'ssd', 'flashdisk', 'memory', 'storage', 'processor', 'gpu',
    
    # Elektronik Rumah Tangga
    'tv', 'televisi', 'kulkas', 'lemari es', 'ac', 'kipas', 'fan', 'blender', 'mixer',
    'rice cooker', 'magic com', 'dispenser', 'setrika', 'iron', 'vacuum', 'mesin cuci',
    'washing machine', 'microwave', 'oven', 'kompor', 'stove',
    
    # Audio Video
    'kamera', 'camera', 'webcam', 'dashcam', 'cctv', 'drone', 'gimbal', 'tripod',
    'lensa', 'lens', 'flash', 'lighting',
    
    # Gaming & Entertainment
    'playstation', 'ps5', 'ps4', 'xbox', 'nintendo', 'switch', 'controller', 'joystick',
    'gaming', 'game', 'console',
    
    # Smartwatch & Wearable
    'smartwatch', 'jam tangan', 'fitness tracker', 'band', 'wearable',
    
    # Networking
    'router', 'modem', 'wifi', 'repeater', 'extender', 'access point',
    
    # Tablet & E-Reader
    'tablet', 'ipad', 'kindle', 'e-reader',
    
    # Printer & Scanner
    'printer', 'scanner', 'toner', 'cartridge', 'tinta', 'ink',
    
    # General Electronics Terms
    'elektronik', 'gadget', 'device', 'perangkat', 'alat', 'teknologi', 'digital'
}

# Non-electronics keywords (akan di-reject)
NON_ELECTRONICS_KEYWORDS = {
    # Makanan
    'tomat', 'sayur', 'buah', 'nasi', 'mie', 'roti', 'kue', 'makanan', 'minuman',
    'daging', 'ikan', 'ayam', 'telur', 'susu', 'keju', 'mentega', 'gula', 'garam',
    'sambal', 'saus', 'kopi', 'teh', 'jus', 'snack', 'cemilan',
    
    # Fashion
    'baju', 'celana', 'kemeja', 'kaos', 'dress', 'rok', 'sepatu', 'sandal', 'tas',
    'dompet', 'topi', 'kacamata', 'jam', 'gelang', 'kalung', 'cincin',
    
    # Kosmetik & Perawatan
    'lipstik', 'bedak', 'foundation', 'maskara', 'parfum', 'sabun', 'shampoo',
    'conditioner', 'lotion', 'cream', 'serum', 'skincare', 'makeup',
    
    # Furniture
    'meja', 'kursi', 'lemari', 'kasur', 'bantal', 'guling', 'sprei', 'gorden',
    'karpet', 'sofa', 'rak',
    
    # Buku & Alat Tulis
    'buku', 'novel', 'komik', 'majalah', 'pulpen', 'pensil', 'penghapus', 'penggaris',
    
    # Mainan
    'boneka', 'robot', 'lego', 'puzzle', 'mainan',
    
    # Otomotif (non-electronics)
    'mobil', 'motor', 'sepeda', 'ban', 'oli', 'helm',
    
    # Kesehatan
    'obat', 'vitamin', 'suplemen', 'masker', 'hand sanitizer'
}

# MODEL ARCHITECTURE
class IndoBERTBiLSTM(nn.Module):
    def __init__(self, bert_model_name, num_classes=5, lstm_hidden_dim=256, 
                 lstm_layers=2, dropout=0.3):
        super(IndoBERTBiLSTM, self).__init__()
        
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.bert_hidden_size = self.bert.config.hidden_size
        
        self.bilstm = nn.LSTM(
            input_size=self.bert_hidden_size,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden_dim * 2, num_classes)
        
    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        sequence_output = bert_output.last_hidden_state
        lstm_output, (hidden, cell) = self.bilstm(sequence_output)
        
        mask_expanded = attention_mask.unsqueeze(-1).expand(lstm_output.size()).float()
        sum_embeddings = torch.sum(lstm_output * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        pooled_output = sum_embeddings / sum_mask
        
        pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)
        
        return logits

# PREPROCESSING
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# CONTENT FILTER
def is_electronics_review(text):
    """
    Check if the review is about electronics products
    Returns: (is_valid, message, detected_keywords)
    """
    text_lower = text.lower()
    
    # Tokenize text
    words = re.findall(r'\b\w+\b', text_lower)
    
    # Check for non-electronics keywords
    detected_non_electronics = []
    for word in words:
        if word in NON_ELECTRONICS_KEYWORDS:
            detected_non_electronics.append(word)
    
    if detected_non_electronics:
        return False, f"Ulasan mengandung kata non-elektronik: {', '.join(set(detected_non_electronics))}", []
    
    # Check for electronics keywords
    detected_electronics = []
    for word in words:
        if word in ELECTRONICS_KEYWORDS:
            detected_electronics.append(word)
    
    # If contains electronics keywords, accept
    if detected_electronics:
        return True, "Ulasan valid tentang produk elektronik", detected_electronics
    
    # If no specific keywords but general review (about delivery, service, etc.), accept
    general_review_keywords = [
        'pengiriman', 'delivery', 'packing', 'packaging', 'kemasan', 'box', 'dus',
        'pelayanan', 'service', 'respon', 'cepat', 'lama', 'bagus', 'jelek',
        'recommend', 'recommended', 'puas', 'kecewa', 'bintang', 'rating',
        'kualitas', 'harga', 'murah', 'mahal', 'ori', 'original', 'palsu', 'fake'
    ]
    
    has_general = any(keyword in text_lower for keyword in general_review_keywords)
    
    if has_general and len(words) >= 3:
        return True, "Ulasan umum tentang layanan/kualitas", []
    
    # Very short text or unclear content
    if len(words) < 3:
        return False, "Ulasan terlalu pendek atau tidak jelas", []
    
    # Default: if no clear non-electronics words, allow it
    return True, "Ulasan tidak mengandung kata terlarang", []

# LOAD MODEL
@st.cache_resource
def load_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = IndoBERTBiLSTM(
        bert_model_name='indolem/indobert-base-uncased',
        num_classes=5,
        lstm_hidden_dim=256,
        lstm_layers=2,
        dropout=0.3
    )
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, device

@st.cache_resource
def load_tokenizer():
    return BertTokenizer.from_pretrained('indolem/indobert-base-uncased')

# PREDICTION
def predict_emotion(text, model, tokenizer, device, max_len=128):
    cleaned_text = preprocess_text(text)
    
    encoding = tokenizer.encode_plus(
        cleaned_text,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, dim=1)
    
    emotion_mapping = {0: 'Marah', 1: 'Kecewa', 2: 'Bingung', 3: 'Senang', 4: 'Normal'}
    urgency_mapping = {0: 3, 1: 3, 2: 2, 3: 1, 4: 1}
    
    predicted_label = predicted.item()
    emotion = emotion_mapping[predicted_label]
    urgency = urgency_mapping[predicted_label]
    confidence_score = confidence.item()
    
    all_probs = {emotion_mapping[i]: probabilities[0][i].item() for i in range(5)}
    
    return {
        'emotion': emotion,
        'urgency': urgency,
        'confidence': confidence_score,
        'probabilities': all_probs,
        'cleaned_text': cleaned_text
    }

# MAIN APP
def main():
    # Header
    st.markdown('<h1 class="main-header">SISTEM KLASIFIKASI EMOSI ULASAN PRODUK ELEKTRONIK</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Analisis Emosi dan Tingkat Urgensi Berbasis IndoBERT-BiLSTM</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2991/2991148.png", width=100)
        st.title("Pengaturan Sistem")
        
        model_option = st.selectbox(
            "Pilih Model:",
            ["best_model_70_30.pth", "best_model_80_20.pth"],
            help="Pilih model yang telah dilatih"
        )
        
        st.markdown("---")
        
        st.markdown("### Informasi Model")
        st.info("""
        **Arsitektur:**
        - IndoBERT (base-uncased)
        - BiLSTM (2 layers, 256 hidden)
        - Fully Connected Layer
        
        **Klasifikasi Emosi:**
        - Marah (Urgensi: 3)
        - Kecewa (Urgensi: 3)
        - Bingung (Urgensi: 2)
        - Senang (Urgensi: 1)
        - Normal (Urgensi: 1)
        """)
        
        st.markdown("---")
        
        st.markdown("### Panduan Penggunaan")
        st.success("""
        1. Pilih mode analisis (Single/Batch)
        2. Input ulasan produk elektronik
        3. Sistem akan memvalidasi konten
        4. Lihat hasil prediksi emosi
        """)
        
        st.markdown("---")
        
        st.markdown("### Filter Konten")
        st.warning("""
        Sistem hanya menerima ulasan tentang:
        - Produk elektronik
        - Layanan umum (pengiriman, dll)
        
        Ulasan non-elektronik akan ditolak.
        """)
    
    # Load model
    try:
        with st.spinner('Memuat model...'):
            model, device = load_model(model_option)
            tokenizer = load_tokenizer()
        st.sidebar.success(f"Model berhasil dimuat: {model_option}")
    except Exception as e:
        st.error(f"Error memuat model: {str(e)}")
        st.stop()
    
    # Mode selection
    st.markdown("---")
    mode = st.radio(
        "Pilih Mode Analisis:",
        ["Analisis Single Text", "Analisis Batch (CSV)"],
        horizontal=True
    )
    
    if mode == "Analisis Single Text":
        single_text_analysis(model, tokenizer, device)
    else:
        batch_csv_analysis(model, tokenizer, device)

def single_text_analysis(model, tokenizer, device):
    st.markdown("## Analisis Single Text")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Input Ulasan")
        input_text = st.text_area(
            "Masukkan ulasan produk elektronik:",
            height=150,
            placeholder="Contoh: Laptop ini sangat bagus, performa cepat dan layar jernih. Sangat puas dengan pembelian ini!",
            help="Masukkan ulasan dalam Bahasa Indonesia"
        )
        
        if st.button("Analisis Emosi", type="primary"):
            if input_text.strip():
                # Validate content
                is_valid, message, keywords = is_electronics_review(input_text)
                
                if not is_valid:
                    st.markdown(f'<div class="error-box"><strong>Ulasan Ditolak!</strong><br>{message}</div>', unsafe_allow_html=True)
                    st.error("Sistem hanya menerima ulasan produk elektronik. Silakan input ulasan yang sesuai.")
                else:
                    # Show validation success
                    if keywords:
                        st.markdown(f'<div class="success-box"><strong>Validasi Berhasil!</strong><br>{message}<br>Kata kunci terdeteksi: {", ".join(list(set(keywords))[:5])}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="info-box"><strong>Validasi Berhasil!</strong><br>{message}</div>', unsafe_allow_html=True)
                    
                    # Predict
                    with st.spinner('Menganalisis emosi...'):
                        result = predict_emotion(input_text, model, tokenizer, device)
                    
                    # Display results
                    st.markdown("---")
                    st.markdown("## Hasil Analisis")
                    
                    # Metrics
                    col_a, col_b, col_c = st.columns(3)
                    
                    emotion_emoji = {
                        'Marah': '😡', 'Kecewa': '😞', 'Bingung': '😕',
                        'Senang': '😊', 'Normal': '😐'
                    }
                    
                    with col_a:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">EMOSI TERDETEKSI</div>
                            <div class="metric-value">{emotion_emoji[result['emotion']]} {result['emotion']}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_b:
                        urgency_emoji = {3: '🔴', 2: '🟡', 1: '🟢'}
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">TINGKAT URGENSI</div>
                            <div class="metric-value">{urgency_emoji[result['urgency']]} Level {result['urgency']}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_c:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">CONFIDENCE SCORE</div>
                            <div class="metric-value">{result['confidence']:.2%}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    # Probability distribution
                    col_chart1, col_chart2 = st.columns(2)
                    
                    with col_chart1:
                        st.markdown("### Distribusi Probabilitas")
                        prob_df = pd.DataFrame({
                            'Emosi': list(result['probabilities'].keys()),
                            'Probabilitas': list(result['probabilities'].values())
                        })
                        
                        fig = px.bar(
                            prob_df,
                            x='Emosi',
                            y='Probabilitas',
                            color='Probabilitas',
                            color_continuous_scale='Viridis',
                            text='Probabilitas'
                        )
                        fig.update_traces(texttemplate='%{text:.2%}', textposition='outside')
                        fig.update_layout(showlegend=False, height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col_chart2:
                        st.markdown("### Perbandingan Emosi")
                        fig = go.Figure(data=[go.Pie(
                            labels=list(result['probabilities'].keys()),
                            values=list(result['probabilities'].values()),
                            hole=.3,
                            marker=dict(colors=['#FF6B6B', '#FFA500', '#FFD700', '#4ECDC4', '#95E1D3'])
                        )])
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Interpretation
                    st.markdown("---")
                    st.markdown("### Interpretasi Hasil")
                    
                    if result['urgency'] == 3:
                        st.markdown(f"""
                        <div class="error-box">
                            <strong>PERHATIAN TINGGI DIPERLUKAN!</strong><br><br>
                            Ulasan menunjukkan emosi <strong>{result['emotion']}</strong> dengan tingkat urgensi <strong>Level 3</strong>.<br>
                            Tindakan: Segera tindaklanjuti feedback pelanggan. Hubungi customer service atau tim terkait untuk penanganan prioritas.
                        </div>
                        """, unsafe_allow_html=True)
                    elif result['urgency'] == 2:
                        st.markdown(f"""
                        <div class="warning-box">
                            <strong>PERLU PERHATIAN</strong><br><br>
                            Ulasan menunjukkan emosi <strong>{result['emotion']}</strong> dengan tingkat urgensi <strong>Level 2</strong>.<br>
                            Tindakan: Pelanggan mungkin memerlukan klarifikasi atau bantuan. Berikan informasi yang jelas dan helpful.
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="success-box">
                            <strong>STATUS NORMAL</strong><br><br>
                            Ulasan menunjukkan emosi <strong>{result['emotion']}</strong> dengan tingkat urgensi <strong>Level 1</strong>.<br>
                            Tindakan: Pelanggan dalam kondisi baik. Pertahankan kualitas layanan.
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Additional info
                    with st.expander("Informasi Preprocessing"):
                        st.write("**Teks Asli:**")
                        st.text(input_text)
                        st.write("**Teks Setelah Preprocessing:**")
                        st.text(result['cleaned_text'])
            else:
                st.warning("Mohon masukkan teks ulasan terlebih dahulu.")
    
    
def batch_csv_analysis(model, tokenizer, device):
    st.markdown("## Analisis Batch (CSV)")
    
    st.markdown("""
    <div class="upload-section">
        <h3>Upload File CSV</h3>
        <p>Format CSV harus memiliki kolom: <strong>text</strong> atau <strong>ulasan</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Drag & Drop atau Browse File CSV",
        type=['csv'],
        help="Upload file CSV yang berisi kolom 'text' atau 'ulasan'"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Check columns
            if 'text' in df.columns:
                text_column = 'text'
            elif 'ulasan' in df.columns:
                text_column = 'ulasan'
            else:
                st.error("CSV harus memiliki kolom 'text' atau 'ulasan'")
                return
            
            st.success(f"File berhasil dimuat! Total data: {len(df)}")
            
            # Preview data
            with st.expander("Preview Data (5 baris pertama)"):
                st.dataframe(df.head(), use_container_width=True)
            
            # Process button
            if st.button("Proses Semua Data", type="primary"):
                results = []
                valid_count = 0
                invalid_count = 0
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, row in df.iterrows():
                    progress = (idx + 1) / len(df)
                    progress_bar.progress(progress)
                    status_text.text(f"Memproses: {idx + 1}/{len(df)}")
                    
                    text = str(row[text_column])
                    
                    # Validate content
                    is_valid, message, keywords = is_electronics_review(text)
                    
                    if not is_valid:
                        results.append({
                            'No': idx + 1,
                            'Ulasan': text,
                            'Status': 'DITOLAK',
                            'Alasan': message,
                            'Emosi': '-',
                            'Urgensi': '-',
                            'Confidence': '-'
                        })
                        invalid_count += 1
                    else:
                        # Predict
                        result = predict_emotion(text, model, tokenizer, device)
                        results.append({
                            'No': idx + 1,
                            'Ulasan': text,
                            'Status': 'VALID',
                            'Alasan': message,
                            'Emosi': result['emotion'],
                            'Urgensi': result['urgency'],
                            'Confidence': f"{result['confidence']:.2%}"
                        })
                        valid_count += 1
                
                progress_bar.empty()
                status_text.empty()
                
                # Create results dataframe
                results_df = pd.DataFrame(results)
                
                # Summary
                st.markdown("---")
                st.markdown("## Ringkasan Hasil")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">TOTAL DATA</div>
                        <div class="metric-value">{len(df)}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card" style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);">
                        <div class="metric-label">DATA VALID</div>
                        <div class="metric-value">{valid_count}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-card" style="background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);">
                        <div class="metric-label">DATA DITOLAK</div>
                        <div class="metric-value">{invalid_count}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Visualization
                if valid_count > 0:
                    valid_results = results_df[results_df['Status'] == 'VALID']
                    
                    col_viz1, col_viz2 = st.columns(2)
                    
                    with col_viz1:
                        st.markdown("### Distribusi Emosi")
                        emotion_counts = valid_results['Emosi'].value_counts()
                        fig = px.pie(
                            values=emotion_counts.values,
                            names=emotion_counts.index,
                            color=emotion_counts.index,
                            color_discrete_map={
                                'Marah': '#FF6B6B',
                                'Kecewa': '#FFA500',
                                'Bingung': '#FFD700',
                                'Senang': '#4ECDC4',
                                'Normal': '#95E1D3'
                            }
                        )
                        fig.update_traces(textposition='inside', textinfo='percent+label')
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col_viz2:
                        st.markdown("### Distribusi Urgensi")
                        urgency_counts = valid_results['Urgensi'].value_counts().sort_index()
                        fig = px.bar(
                            x=urgency_counts.index.astype(str),
                            y=urgency_counts.values,
                            labels={'x': 'Tingkat Urgensi', 'y': 'Jumlah'},
                            color=urgency_counts.values,
                            color_continuous_scale=['green', 'yellow', 'red']
                        )
                        fig.update_layout(showlegend=False, height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Emotion statistics
                    st.markdown("### Statistik Detail")
                    col_stat1, col_stat2, col_stat3 = st.columns(3)
                    
                    with col_stat1:
                        urgent_high = len(valid_results[valid_results['Urgensi'] == 3])
                        st.metric("Urgensi Tinggi (Level 3)", urgent_high, 
                                 delta=f"{urgent_high/valid_count*100:.1f}%" if valid_count > 0 else "0%",
                                 delta_color="inverse")
                    
                    with col_stat2:
                        urgent_med = len(valid_results[valid_results['Urgensi'] == 2])
                        st.metric("Urgensi Sedang (Level 2)", urgent_med,
                                 delta=f"{urgent_med/valid_count*100:.1f}%" if valid_count > 0 else "0%",
                                 delta_color="off")
                    
                    with col_stat3:
                        urgent_low = len(valid_results[valid_results['Urgensi'] == 1])
                        st.metric("Urgensi Rendah (Level 1)", urgent_low,
                                 delta=f"{urgent_low/valid_count*100:.1f}%" if valid_count > 0 else "0%",
                                 delta_color="normal")
                
                st.markdown("---")
                
                # Display results table
                st.markdown("### Hasil Lengkap")
                
                # Filter options
                col_filter1, col_filter2 = st.columns(2)
                
                with col_filter1:
                    status_filter = st.multiselect(
                        "Filter Status:",
                        options=['VALID', 'DITOLAK'],
                        default=['VALID', 'DITOLAK']
                    )
                
                with col_filter2:
                    if valid_count > 0:
                        emotion_filter = st.multiselect(
                            "Filter Emosi:",
                            options=['Marah', 'Kecewa', 'Bingung', 'Senang', 'Normal'],
                            default=['Marah', 'Kecewa', 'Bingung', 'Senang', 'Normal']
                        )
                    else:
                        emotion_filter = []
                
                # Apply filters
                filtered_df = results_df[results_df['Status'].isin(status_filter)]
                if valid_count > 0 and emotion_filter:
                    filtered_df = filtered_df[
                        (filtered_df['Status'] == 'DITOLAK') | 
                        (filtered_df['Emosi'].isin(emotion_filter))
                    ]
                
                # Style the dataframe
                def highlight_status(row):
                    if row['Status'] == 'DITOLAK':
                        return ['background-color: #ffcccc'] * len(row)
                    elif row['Urgensi'] == 3:
                        return ['background-color: #ffe6e6'] * len(row)
                    elif row['Urgensi'] == 2:
                        return ['background-color: #fff9e6'] * len(row)
                    else:
                        return ['background-color: #e6ffe6'] * len(row)
                
                st.dataframe(
                    filtered_df.style.apply(highlight_status, axis=1),
                    use_container_width=True,
                    height=400
                )
                
                # Download results
                st.markdown("### Download Hasil")
                
                col_dl1, col_dl2 = st.columns(2)
                
                with col_dl1:
                    # CSV download
                    csv = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Full Results (CSV)",
                        data=csv,
                        file_name=f"emotion_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                    )
                
                with col_dl2:
                    # Valid data only
                    if valid_count > 0:
                        valid_csv = results_df[results_df['Status'] == 'VALID'].to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Valid Data Only (CSV)",
                            data=valid_csv,
                            file_name=f"emotion_analysis_valid_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                        )
                
                # Priority recommendations
                if valid_count > 0:
                    st.markdown("---")
                    st.markdown("### Rekomendasi Tindakan")
                    
                    high_priority = results_df[
                        (results_df['Status'] == 'VALID') & 
                        (results_df['Urgensi'] == 3)
                    ]
                    
                    if len(high_priority) > 0:
                        st.markdown(f"""
                        <div class="error-box">
                            <strong>PERHATIAN!</strong><br>
                            Terdapat <strong>{len(high_priority)}</strong> ulasan dengan urgensi tinggi yang memerlukan tindakan segera.<br>
                            Segera tindaklanjuti untuk menjaga kepuasan pelanggan.
                        </div>
                        """, unsafe_allow_html=True)
                        
                        with st.expander("Lihat Ulasan Prioritas Tinggi"):
                            st.dataframe(
                                high_priority[['No', 'Ulasan', 'Emosi', 'Confidence']],
                                use_container_width=True
                            )
                    else:
                        st.markdown("""
                        <div class="success-box">
                            <strong>STATUS BAIK</strong><br>
                            Tidak ada ulasan dengan urgensi tinggi. Pertahankan kualitas layanan!
                        </div>
                        """, unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"Error memproses file: {str(e)}")
            st.info("Pastikan file CSV memiliki kolom 'text' atau 'ulasan' dan format yang benar.")

if __name__ == "__main__":
    main()