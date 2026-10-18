from pypdf import PdfReader

with open("archives/Group_28_Report_Final.pdf", "rb") as f:
    reader = PdfReader(f)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"

with open("report_text.txt", "w") as f:
    f.write(text)
print("Extracted to report_text.txt")
