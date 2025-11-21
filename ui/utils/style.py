page_style = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=Lato:wght@300;400;700&display=swap');

    /* Global Font Settings */
    html, body, [class*="css"] {
        font-family: 'Lato', sans-serif;
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Playfair Display', serif;
        color: #000000 !important;
        font-weight: 600;
    }

    /* Banner Styling */
    .main-header {
        text-align: center;
        margin-bottom: 3rem;
    }
    
    /* Tabs Styling - Minimalist */
    .stTabs [data-baseweb="tab-list"] {
        gap: 40px;
        border-bottom: 1px solid #eee;
        padding-bottom: 10px;
    }

    .stTabs [data-baseweb="tab"] {
        height: auto;
        background-color: transparent;
        border: none;
        padding: 0;
        font-family: 'Playfair Display', serif;
        font-size: 1.2rem;
        color: #999;
    }

    .stTabs [aria-selected="true"] {
        background-color: transparent;
        color: #000 !important;
        font-weight: 700;
        border-bottom: 2px solid #000;
    }

    /* Input Area - Clean */
    .stTextArea textarea {
        font-size: 16px;
        border-radius: 0px;
        border: 1px solid #ddd;
        background-color: #fff;
        color: #000;
        padding: 15px;
    }
    .stTextArea textarea:focus {
        border-color: #000;
        box-shadow: none;
    }

    /* Buttons - Minimal */
    .stButton button {
        border-radius: 0px;
        font-weight: 400;
        padding: 0.6rem 2rem;
        background-color: #000;
        color: #fff;
        border: 1px solid #000;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-size: 0.8rem;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        background-color: #fff;
        color: #000;
        border: 1px solid #000;
    }
    
    /* Remove default top padding */
    .block-container { padding-top: 2rem; padding-bottom: 5rem; }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #fcfcfc;
        border-right: 1px solid #eee;
    }
    
    </style>
"""