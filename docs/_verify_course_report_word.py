import os

import win32com.client as win32


PATH = os.environ["COURSE_REPORT_DOC_PATH"]
WD_STAT_PAGES = 2

checks = [
    "摘要",
    "第一章 绪论",
    "第二章 环境配置与模型选择",
    "第三章 系统设计与实现",
    "第四章 实验与评测",
    "第五章 总结与展望",
    "参考文献",
    "附录",
    "表4.1 单模型 full30 评测结果",
    "表4.2 RAG 策略优化对比结果",
    "表4.3 Prompt 策略对比结果",
    "表4.4 双模型对比结果",
    "表4.5 检索权重调优结果",
    "hybrid 相对 vector 将检索命中率从 0.7000 提升到 1.0000",
    "deepseek-r1:7b-qwen-distill-q4_K_M",
    "qwen2.5:7b-instruct-q4_K_M",
    "http://127.0.0.1:8090",
    "document_count=41",
    "chunk_count=163",
    "full30_20260621 是本报告使用的主实验结果",
]

word = win32.Dispatch("Word.Application")
word.Visible = False
word.DisplayAlerts = 0
doc = word.Documents.Open(PATH)
try:
    text = doc.Content.Text
    print("pages", doc.ComputeStatistics(WD_STAT_PAGES))
    print("paragraphs", doc.Paragraphs.Count)
    print("tables", doc.Tables.Count)
    for needle in checks:
        print(("OK " if needle in text else "MISS ") + needle)
finally:
    doc.Close(False)
    word.Quit()
