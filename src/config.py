import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

categorical_features = [
    "Age",
    "Gender",
    "Ethnicity",
    "ParentalEducation",
    "Tutoring",
    "ParentalInvolvement",
    "Extracurricular",
    "Sports",
    "Music",
    "Volunteering",
    "GradeClass",
]
numerical_features = ["StudyTimeWeekly", "Absences", "GPA"]
font_path = "assets/fonts/PingFang-Medium.ttf"  # 替换为你系统中存在的中文字体文件路径
font = fm.FontProperties(fname=font_path)
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示为方块的问题
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 用来正常显示中文标签
DATA_PATH = "./assets/Student_performance_data.csv"
