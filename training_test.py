import spacy
from spacy.matcher import PhraseMatcher


sentence = "Communication, Verbal Communication,Body Language,Physical Communication,Writing,Storytelling,Visual Communication,Humor,Quick-wittedness,Listening,Presentation Skills,Public Speaking,Interviewing,Leadership,Team Building,Strategic Planning,Coaching,Mentoring,Delegation,Dispute Resolution,Diplomacy,Giving Feedback,Managing Difficult Conversations,Decision Making,Performance Management,Supervising,Managing,Manager Management,Talent Management,Managing Remote Teams,Managing Virtual Teams,Crisis Management,Influencing,Facilitation,Selling,Inspiring,Persuasion,Negotiation,Motivating,Collaborating,Interpersonal Skills,Networking,Interpersonal Relationships,Dealing with Difficult People,Conflict Resolution,Personal Branding,Office Politics,Personal Skills,Emotional Intelligence,Self Awareness,Emotion Management,Stress Management,Tolerance of Change and Uncertainty,Taking Criticism,Self Confidence,Adaptability,Resilience,Assertiveness,Competitiveness,Self Leadership,Self Assessment,Work-Life Balance,Friendliness,Enthusiasm,Empathy,Creativity,Problem Solving,Critical Thinking,Innovation,Troubleshooting,Design Sense,Artistic Sense,Professional Skills,Organization,Planning,Scheduling,Time Management,Meeting Management,Technology Savvy,Technology Trend Awareness,Business Trend Awareness,Research,Business Etiquette,Business Ethics,Diversity Awareness,Disability Awareness,Intercultural Competence,Training,Train the Trainer,Process Improvement,Knowledge Management,Writing Reports and Proposals,Customer Service,Entrepreneurial Thinking "
terms = sentence.split(",")


nlp = spacy.load('en_core_web_sm')
matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
# terms = ["Barack Obama", "Angela Merkel", "Washington, D.C."]
# Only run nlp.make_doc to speed things up
patterns = [nlp.make_doc(text) for text in terms]
matcher.add("SoftSkillList", None, *patterns)

doc = nlp("Key Strengths: Effective Communication Skills and Zeal to learn. Flexibility and Adaptability. Good Leadership Qualities. Analytical and Problem Solving Skills")
matches = matcher(doc)
for match_id, start, end in matches:
    span = doc[start:end]
    print(span.text)

