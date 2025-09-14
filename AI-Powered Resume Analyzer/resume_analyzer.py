import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import PyPDF2
import docx
from io import BytesIO
import time
from datetime import datetime
import base64

# Download NLTK data if not present
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('corpora/stopwords')
except LookupError:
    st.info("📥 İlk kullanım: NLTK data indiriliyor...")
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    st.success("✅ NLTK data başarıyla indirildi!")

# Page configuration
st.set_page_config(
    page_title="AI-Powered Resume Analyzer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin: 0.5rem 0;
    }
    
    .improvement-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        color: white;
    }
    
    .stProgress .st-bo {
        background-color: #667eea;
    }
    
    .upload-section {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background: rgba(102, 126, 234, 0.1);
    }
</style>
""", unsafe_allow_html=True)

class ResumeAnalyzer:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        
        # Define skill categories and keywords
        self.skill_categories = {
            'Technical Skills': [
                'python', 'java', 'javascript', 'react', 'angular', 'vue', 'nodejs', 'django',
                'flask', 'sql', 'mongodb', 'postgresql', 'mysql', 'docker', 'kubernetes',
                'aws', 'azure', 'gcp', 'git', 'linux', 'html', 'css', 'machine learning',
                'artificial intelligence', 'data science', 'tensorflow', 'pytorch', 'pandas',
                'numpy', 'scikit-learn', 'api', 'rest', 'graphql', 'microservices'
            ],
            'Soft Skills': [
                'leadership', 'communication', 'teamwork', 'problem solving', 'analytical',
                'creative', 'innovative', 'collaborative', 'adaptable', 'organized',
                'detail-oriented', 'time management', 'project management', 'mentoring',
                'strategic thinking', 'decision making', 'conflict resolution'
            ],
            'Certifications': [
                'certified', 'certification', 'aws certified', 'pmp', 'scrum master',
                'agile', 'comptia', 'cissp', 'cisa', 'cism', 'google certified',
                'microsoft certified', 'oracle certified', 'cisco certified'
            ]
        }
        
        # ATS-friendly keywords
        self.ats_keywords = [
            'experience', 'skills', 'education', 'projects', 'achievements',
            'responsibilities', 'managed', 'developed', 'implemented', 'created',
            'improved', 'increased', 'decreased', 'led', 'collaborated', 'designed'
        ]

    def extract_text_from_pdf(self, file):
        """Extract text from PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
        except Exception as e:
            st.error(f"Error reading PDF: {str(e)}")
            return ""

    def extract_text_from_docx(self, file):
        """Extract text from DOCX file"""
        try:
            doc = docx.Document(file)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            st.error(f"Error reading DOCX: {str(e)}")
            return ""

    def preprocess_text(self, text):
        """Clean and preprocess text"""
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text

    def extract_skills(self, text):
        """Extract skills from resume text"""
        found_skills = {'Technical Skills': [], 'Soft Skills': [], 'Certifications': []}
        
        for category, skills in self.skill_categories.items():
            for skill in skills:
                if skill.lower() in text.lower():
                    found_skills[category].append(skill)
        
        return found_skills

    def calculate_ats_score(self, text):
        """Calculate ATS compatibility score"""
        text_lower = text.lower()
        found_keywords = sum(1 for keyword in self.ats_keywords if keyword in text_lower)
        max_score = len(self.ats_keywords)
        
        # Additional checks
        bonus_points = 0
        
        # Check for contact information
        if re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text):
            bonus_points += 10
        
        # Check for phone number
        if re.search(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text):
            bonus_points += 5
        
        # Check for sections
        sections = ['experience', 'education', 'skills', 'summary', 'objective']
        for section in sections:
            if section in text_lower:
                bonus_points += 3
        
        score = min(100, (found_keywords / max_score) * 70 + bonus_points)
        return score

    def analyze_resume_structure(self, text):
        """Analyze resume structure and format"""
        structure_score = 0
        issues = []
        
        # Check length (optimal: 1-2 pages, ~300-600 words)
        word_count = len(text.split())
        if 300 <= word_count <= 800:
            structure_score += 25
        elif word_count < 300:
            issues.append("Resume appears too short. Consider adding more details.")
        else:
            issues.append("Resume appears too long. Consider condensing information.")
        
        # Check for key sections
        required_sections = ['experience', 'education', 'skills']
        found_sections = sum(1 for section in required_sections if section in text.lower())
        structure_score += (found_sections / len(required_sections)) * 25
        
        # Check for contact info
        has_email = bool(re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text))
        has_phone = bool(re.search(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text))
        
        if has_email and has_phone:
            structure_score += 25
        elif has_email or has_phone:
            structure_score += 15
            issues.append("Missing contact information (email or phone).")
        else:
            issues.append("No contact information found.")
        
        # Check for quantifiable achievements
        numbers = re.findall(r'\d+%|\$\d+|\d+\+|increase.*\d+|decrease.*\d+', text.lower())
        if len(numbers) >= 3:
            structure_score += 25
        elif len(numbers) >= 1:
            structure_score += 15
        else:
            issues.append("Add quantifiable achievements (percentages, numbers, metrics).")
        
        return min(structure_score, 100), issues

    def generate_recommendations(self, skills, ats_score, structure_issues):
        """Generate personalized recommendations"""
        recommendations = []
        
        # Skill-based recommendations
        if len(skills['Technical Skills']) < 5:
            recommendations.append("💡 Add more technical skills relevant to your field")
        
        if len(skills['Soft Skills']) < 3:
            recommendations.append("🤝 Include more soft skills to show your interpersonal abilities")
        
        # ATS recommendations
        if ats_score < 70:
            recommendations.append("🎯 Optimize for ATS by including more industry keywords")
            recommendations.append("📝 Use standard section headers (Experience, Education, Skills)")
        
        # Structure recommendations
        recommendations.extend([f"📋 {issue}" for issue in structure_issues[:3]])
        
        # General recommendations
        recommendations.extend([
            "🌟 Use action verbs to start bullet points (Managed, Developed, Created)",
            "📊 Include specific metrics and achievements where possible",
            "🔗 Add relevant certifications or training programs",
            "👀 Ensure consistent formatting throughout the document"
        ])
        
        return recommendations[:8]  # Limit to top 8 recommendations

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 AI-Powered Resume Analyzer</h1>
        <p style="font-size: 1.2em; margin-top: 0.5rem;">
            Get instant AI-driven insights to boost your resume's impact
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize analyzer
    analyzer = ResumeAnalyzer()

    # File upload section
    st.markdown("### 📁 Upload Your Resume")
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['pdf', 'docx', 'txt'],
        help="Supported formats: PDF, DOCX, TXT"
    )

    if uploaded_file is not None:
        # Show file info
        st.success(f"✅ File uploaded: {uploaded_file.name} ({uploaded_file.size} bytes)")
        
        # Extract text based on file type
        with st.spinner("🔍 Analyzing your resume..."):
            if uploaded_file.type == "application/pdf":
                text = analyzer.extract_text_from_pdf(uploaded_file)
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                text = analyzer.extract_text_from_docx(uploaded_file)
            else:
                text = str(uploaded_file.read(), "utf-8")
            
            if text:
                # Preprocess text
                processed_text = analyzer.preprocess_text(text)
                
                # Perform analysis
                skills = analyzer.extract_skills(processed_text)
                ats_score = analyzer.calculate_ats_score(processed_text)
                structure_score, structure_issues = analyzer.analyze_resume_structure(processed_text)
                
                # Calculate overall score
                skill_score = min(100, len(skills['Technical Skills']) * 8 + len(skills['Soft Skills']) * 6 + len(skills['Certifications']) * 10)
                overall_score = (ats_score * 0.3 + structure_score * 0.4 + skill_score * 0.3)
                
                # Add some animation delay
                time.sleep(1)

        # Results section
        st.markdown("---")
        st.markdown("## 📊 Analysis Results")

        # Overall score with gauge chart
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = overall_score,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Overall Resume Score"},
                delta = {'reference': 75},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig_gauge.update_layout(height=400)
            st.plotly_chart(fig_gauge, use_container_width=True)

        # Category scores
        st.markdown("### 📈 Category Breakdown")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="🎯 ATS Compatibility",
                value=f"{ats_score:.0f}/100",
                delta=f"{ats_score-75:.0f}" if ats_score >= 75 else f"{ats_score-75:.0f}"
            )
        
        with col2:
            st.metric(
                label="📋 Structure Quality",
                value=f"{structure_score:.0f}/100",
                delta=f"{structure_score-80:.0f}" if structure_score >= 80 else f"{structure_score-80:.0f}"
            )
        
        with col3:
            st.metric(
                label="🛠️ Skills Coverage",
                value=f"{skill_score:.0f}/100",
                delta=f"{skill_score-70:.0f}" if skill_score >= 70 else f"{skill_score-70:.0f}"
            )
        
        with col4:
            total_skills = sum(len(skills[cat]) for cat in skills)
            st.metric(
                label="📊 Total Skills",
                value=f"{total_skills}",
                delta=f"{total_skills-15}" if total_skills >= 15 else f"{total_skills-15}"
            )

        # Skills breakdown chart
        st.markdown("### 🛠️ Skills Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Skills by category
            skill_counts = [len(skills[cat]) for cat in skills.keys()]
            fig_skills = px.bar(
                x=list(skills.keys()),
                y=skill_counts,
                title="Skills by Category",
                color=skill_counts,
                color_continuous_scale="viridis"
            )
            fig_skills.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig_skills, use_container_width=True)
        
        with col2:
            # Skills details
            for category, skill_list in skills.items():
                if skill_list:
                    st.markdown(f"**{category}:**")
                    for skill in skill_list[:5]:  # Show top 5
                        st.markdown(f"• {skill.title()}")
                    if len(skill_list) > 5:
                        st.markdown(f"• ... and {len(skill_list) - 5} more")
                    st.markdown("")

        # Recommendations section
        st.markdown("### 💡 AI Recommendations")
        
        recommendations = analyzer.generate_recommendations(skills, ats_score, structure_issues)
        
        for i, rec in enumerate(recommendations, 1):
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 15px; border-radius: 10px; margin: 10px 0; color: white;">
                <strong>{i}.</strong> {rec}
            </div>
            """, unsafe_allow_html=True)

        # Word cloud of resume content
        st.markdown("### ☁️ Resume Word Cloud")
        
        # Simple word frequency analysis
        words = word_tokenize(processed_text.lower())
        words = [word for word in words if word.isalpha() and word not in analyzer.stop_words and len(word) > 2]
        word_freq = Counter(words).most_common(20)
        
        if word_freq:
            df_words = pd.DataFrame(word_freq, columns=['Word', 'Frequency'])
            fig_words = px.bar(df_words, x='Frequency', y='Word', orientation='h',
                             title="Top 20 Most Frequent Words",
                             color='Frequency', color_continuous_scale="plasma")
            fig_words.update_layout(height=600)
            st.plotly_chart(fig_words, use_container_width=True)

        # Export results option
        st.markdown("### 📥 Export Analysis")
        
        if st.button("📊 Download Analysis Report"):
            # Create a simple report
            report_data = {
                'Metric': ['Overall Score', 'ATS Compatibility', 'Structure Quality', 'Skills Coverage', 'Total Skills Found'],
                'Score': [f"{overall_score:.1f}/100", f"{ats_score:.1f}/100", f"{structure_score:.1f}/100", 
                         f"{skill_score:.1f}/100", f"{total_skills}"]
            }
            df_report = pd.DataFrame(report_data)
            
            # Convert to CSV
            csv = df_report.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            
            st.markdown(f"""
            <a href="data:file/csv;base64,{b64}" download="resume_analysis_report.csv">
                <button style="background: #667eea; color: white; padding: 10px 20px; 
                              border: none; border-radius: 5px; cursor: pointer;">
                    📥 Download CSV Report
                </button>
            </a>
            """, unsafe_allow_html=True)

    else:
        # Show demo section
        st.markdown("""
        <div style="text-align: center; padding: 3rem; background: rgba(102, 126, 234, 0.1); 
                    border-radius: 15px; margin: 2rem 0;">
            <h3>🚀 Ready to Boost Your Resume?</h3>
            <p style="font-size: 1.1em; color: #666;">
                Upload your resume above to get instant AI-powered insights including:
            </p>
            <div style="display: flex; justify-content: space-around; margin-top: 2rem; flex-wrap: wrap;">
                <div style="margin: 1rem;">🎯 ATS Compatibility Score</div>
                <div style="margin: 1rem;">📊 Skills Analysis</div>
                <div style="margin: 1rem;">📋 Structure Assessment</div>
                <div style="margin: 1rem;">💡 Personalized Recommendations</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>🤖 AI-Powered Resume Analyzer | Built with Streamlit & Python</p>
        <p style="font-size: 0.9em;">Developed by [Your Name] • 
        <a href="https://github.com/yourusername" target="_blank">GitHub</a> • 
        <a href="https://linkedin.com/in/yourusername" target="_blank">LinkedIn</a></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()