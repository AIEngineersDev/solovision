def get_custom_css():
    return """
    <style>
    /* Reset and base styles */
    .stApp {
        background-color: #0B0425 !important;
    }
    
    .stSidebar {
        background-color: #02010a !important;
    }
    
    /* Hide default header and adjust main container */
    header[data-testid="stHeader"] {
        display: none;
    }
    
    div[data-testid="stAppViewContainer"] > div:first-child {
        padding-top: 1rem !important;
    }
    
    /* Main content container */
    div[data-testid="stAppViewContainer"] {
        background-color: #0B0425 !important;
    }
    
    /* Block container adjustment */
    .block-container {
        padding-top: 1rem !important;
        max-width: 1000px !important;
    }
    
    /* Title section - compact and higher */
    .title-section {
        text-align: center;
        padding: 0.5rem 0;
        margin-bottom: 1.5rem;
    }
    
    .title-section h1 {
        color: white !important;
        font-size: 2.5rem !important;
        font-weight: 600 !important;
        margin-bottom: 0.3rem !important;
        line-height: 1.2 !important;
        margin-top: -0.5rem !important;
    }
    
    .title-section h3 {
        color: rgba(255, 255, 255, 0.8) !important;
        font-size: 1.2rem !important;
        font-weight: 400 !important;
        margin-bottom: 0 !important;
    }
    
    
    /* Description and Features sections - unified styling */

    .description-section, .features-section {
    max-width: 800px;
    margin: 1rem auto;
    padding: 1.5rem;
    border-radius: 8px;
    }

    .description-section {
        margin-top: -3.0rem;
        margin-bottom: 1rem; /* Consistent spacing between sections */
    }

    .features-section {
            margin-top: -4.5rem; /* Reduced overlap and adjusted spacing */
        }

    
    .description-section h2, .features-section h2 {
        color: white !important;
        font-size: 1.5rem !important;
        margin-bottom: 1.0rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.5px !important;
    }

   
    
    .description-section p, .features-section p {
        color: rgba(255, 255, 255, 0.9) !important;
        font-size: 1.1rem !important;
        line-height: 1.7 !important;
        margin-bottom: 1.5rem !important;
        margin-top: -1.0rem;
    }

    /* Features list styling */
    .features-section ul {
        list-style: none !important;
        padding-left: 0 !important;
        margin-top: -1.0rem !important;
    }
    
    .features-section li {
        color: rgba(255, 255, 255, 0.9) !important;
        font-size: 1.1rem !important;
        line-height: 1.7 !important;
        margin-bottom: 0.8rem !important;
        padding-left: 1.5rem !important;
        position: relative !important;
    }
    
    .features-section li:before {
        content: "•" !important;
        color: #4CAF50 !important;
        font-weight: bold !important;
        position: absolute !important;
        left: 0 !important;
    }
    
    /* Sidebar elements */
    .stSidebar [data-testid="stSidebarNav"] {
        padding-top: 0rem !important;
    }
    
    .stSidebar [data-testid="stSidebarContent"] {
        padding-top: 0rem !important;
    }
    
    .stSelectbox select {
        background-color: #02010a !important;
        color: white !important;
    }
    
    .stFileUploader {
        background-color: #02010a !important;
        color: white !important;
    }
    
    /* Streamlit native elements styling */
    .css-10trblm {  /* Subheader style */
        color: white !important;
        font-size: 1.5rem !important;
        margin-top: 1.5rem !important;
        margin-bottom: 1rem !important;
        font-weight: 500 !important;
        padding-left: 1.5rem !important;
    }
    
    .css-q8sbsg p {  /* Bullet points style */
        color: rgba(255, 255, 255, 0.9) !important;
        font-size: 0.95rem !important;
        line-height: 1.6 !important;
        margin-bottom: 0.5rem !important;
    }
    
    .section-header {
        color: white !important;
        font-size: 1.5rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.5px !important;
        margin: 1rem auto !important;
        padding: 1.5rem !important;
        max-width: 800px !important;
    }

    /* Features grid layout */
    .features-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 1.5rem;
        margin-top: 0rem;
    }

    .feature-item {
        background: rgba(255, 255, 255, 0.05);
        padding: 1.5rem 0.8rem !important;;
        border-radius: 8px;
        text-align: center;
    }

    .feature-item h6 {
        color: white !important;
        padding: 0rem !important;
        margin-top
    }

    /* Center the start tracking button */
    .stButton {
        text-align: center !important;
    }

    .stButton > button {
        background-color: #FF4B4B !important;
        color: white !important;
        padding: 0.5rem 1.0rem !important;
        font-size: 1.1rem !important;
        border-radius: 100px !important;
        border: none !important;
        cursor: pointer !important;
        transition: background-color 0.3s !important;
    }

    .stButton > button:hover {
        background-color: #0B0425 !important;
        color: white !important; 
    }
    </style>
    """ 