# ============================================================
# PROMPT ENGINEERING ASSIGNMENT
# All Sections Combined in One File
# ============================================================

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# ------------------------------------------------------------
# 🔹 Initialize Model
# ------------------------------------------------------------
llm = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()


# ============================================================
# 1️⃣ BASIC PROMPT
# Experiment with different input phrases
# ============================================================

basic_template = """
Complete the following sentence:

"{input_phrase}"
"""

basic_prompt = PromptTemplate.from_template(basic_template)
basic_chain = basic_prompt | llm | parser

basic_inputs = [
    "The future of artificial intelligence is",
    "Once upon a time in a distant galaxy",
    "The benefits of sustainable energy include"
]

print("\n========== BASIC PROMPTS ==========")
for phrase in basic_inputs:
    result = basic_chain.invoke({"input_phrase": phrase})
    print("\nInput:", phrase)
    print("Output:", result)


# ============================================================
# 2️⃣ ZERO SHOT PROMPTS
# No examples provided to the model
# ============================================================

print("\n========== ZERO SHOT PROMPTS ==========")

# ---- (a) Movie Review Classification ----
zero_shot_classification = """
Classify the following movie review as Positive or Negative.

Review: {review}

Answer only: Positive or Negative
"""

classification_prompt = PromptTemplate.from_template(zero_shot_classification)
classification_chain = classification_prompt | llm | parser

review = "The movie was absolutely fantastic with brilliant acting."
print("\nMovie Classification:")
print(classification_chain.invoke({"review": review}))


# ---- (b) Summarization ----
zero_shot_summary = """
Summarize the following paragraph in 3 sentences:

{paragraph}
"""

summary_prompt = PromptTemplate.from_template(zero_shot_summary)
summary_chain = summary_prompt | llm | parser

paragraph = """
Climate change refers to long-term shifts in temperature and weather patterns,
mainly caused by human activities such as burning fossil fuels.
"""

print("\nSummarization:")
print(summary_chain.invoke({"paragraph": paragraph}))


# ---- (c) Translation ----
zero_shot_translation = """
Translate the following English text into Spanish:

{sentence}
"""

translation_prompt = PromptTemplate.from_template(zero_shot_translation)
translation_chain = translation_prompt | llm | parser

print("\nTranslation:")
print(translation_chain.invoke({"sentence": "Good morning, how are you?"}))


# ============================================================
# 3️⃣ ONE SHOT PROMPTS
# One example provided before asking new task
# ============================================================

print("\n========== ONE SHOT PROMPTS ==========")

# ---- (a) Formal Email ----
one_shot_email = """
Example:
Subject: Leave Application
Dear Sir,
I am writing to request leave for two days due to personal reasons.
Thank you.
Sincerely,
John

Now write a similar formal email for:
{topic}
"""

email_prompt = PromptTemplate.from_template(one_shot_email)
email_chain = email_prompt | llm | parser

print("\nFormal Email:")
print(email_chain.invoke({"topic": "Request for project deadline extension"}))


# ---- (b) Simple Explanation ----
one_shot_explanation = """
Example:
Technical Concept: Cloud Computing
Simple Explanation: Cloud computing means storing and accessing data over the internet instead of your computer.

Now explain this concept simply:
{concept}
"""

explain_prompt = PromptTemplate.from_template(one_shot_explanation)
explain_chain = explain_prompt | llm | parser

print("\nSimple Explanation:")
print(explain_chain.invoke({"concept": "Blockchain"}))


# ---- (c) Keyword Extraction ----
one_shot_keywords = """
Example:
Sentence: Artificial Intelligence is transforming healthcare and finance.
Keywords: Artificial Intelligence, healthcare, finance

Now extract keywords from:
Sentence: {sentence}

Keywords:
"""

keyword_prompt = PromptTemplate.from_template(one_shot_keywords)
keyword_chain = keyword_prompt | llm | parser

print("\nKeyword Extraction:")
print(keyword_chain.invoke({"sentence": "Cybersecurity protects systems from digital attacks."}))


# ============================================================
# 4️⃣ CHAIN OF THOUGHT PROMPTS
# Encourages step-by-step reasoning
# ============================================================

print("\n========== CHAIN OF THOUGHT PROMPTS ==========")

# ---- (a) Study vs Movie Decision ----
cot_decision = """
A student has an important test in two days.
They are thinking about going to a movie tonight.

Think step-by-step about the pros and cons.
Then give a final recommendation.
"""

cot_prompt = PromptTemplate.from_template(cot_decision)
cot_chain = cot_prompt | llm | parser

print("\nDecision Making:")
print(cot_chain.invoke({}))


# ---- (b) Peanut Butter & Jelly Sandwich ----
cot_sandwich = """
Explain step-by-step how to make a peanut butter and jelly sandwich.
Think carefully and list each step clearly.
"""

sandwich_prompt = PromptTemplate.from_template(cot_sandwich)
sandwich_chain = sandwich_prompt | llm | parser

print("\nSandwich Steps:")
print(sandwich_chain.invoke({}))


# ============================================================
# 5️⃣ TEXT CLASSIFICATION AGENT
# Built similar to Q&A Agent but for classification
# ============================================================

print("\n========== TEXT CLASSIFICATION AGENT ==========")

classification_agent_template = """
Classify the following text into one category:
Technology, Sports, Politics, Health

Text:
{content}

Answer only the category.
"""

agent_prompt = PromptTemplate.from_template(classification_agent_template)
agent_chain = agent_prompt | llm | parser

content = """
Artificial Intelligence is transforming industries by automating tasks
and improving efficiency across businesses.
"""

print("\nCategory:")
print(agent_chain.invoke({"content": content}))
