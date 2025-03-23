import os
import google.generativeai as genai

# Configure Gemini API (if available)
api_key = os.environ.get("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")
else:
    model = None
    print("Warning: GEMINI_API_KEY not set. Using fallback logic.")

def explain_prediction(probability, input_dict, surname):
    """Generate a detailed explanation for churn likelihood, with or without Gemini."""
    feature_importance = {
        "NumOfProducts": 0.323888,
        "IsActiveMember": 0.164146,
        "Age": 0.109558,
        "Geography_Germany": 0.091373,
        "Balance": 0.052786,
        "Geography_France": 0.046463,
        "Gender_Female": 0.045283,
        "Geography_Spain": 0.036055,
        "CreditScore": 0.035085,
        "EstimatedSalary": 0.032655
    }
    
    processed_input = {
        "NumOfProducts": input_dict.get("NumOfProducts", 0),
        "IsActiveMember": input_dict.get("IsActiveMember", 0),
        "Age": input_dict.get("Age", 0),
        f"Geography_{input_dict.get('Geography', 'France')}": 1,
        "Balance": input_dict.get("Balance", 0),
        "Gender_{}".format(input_dict.get("Gender", "Male")): 1,
        "CreditScore": input_dict.get("CreditScore", 0),
        "EstimatedSalary": input_dict.get("EstimatedSalary", 0),
        "HasCrCard": input_dict.get("HasCrCard", 0),
        "Tenure": input_dict.get("Tenure", 0)
    }
    
    summary_stats = {
        "Avg NumOfProducts": 1.8,
        "Avg Age": 45,
        "Avg Balance": 120000,
        "IsActiveMember_pct": 20,
        "Geography_Germany_pct": 40,
        "Avg CreditScore": 620,
        "Avg EstimatedSalary": 110000
    }
    
    risk_threshold = 0.4
    intro = (
        f"{surname} is at risk of leaving due to several factors we’ve noticed."
        if probability > risk_threshold
        else f"{surname} might not be at risk of leaving, thanks to some strong habits we’ve observed."
    )
    
    prompt = (
        f"You are an expert Data Scientist at a bank, specializing in interpreting customer behavior. "
        f"A customer named {surname} has a {round(probability * 100, 1)}% chance of churning, "
        f"based on this information:\n{processed_input}\n\n"
        f"Top 10 features influencing churn:\n"
        f"Feature | Importance\n{'-' * 20}\n"
        f"{'NumOfProducts':<20} | {feature_importance['NumOfProducts']}\n"
        f"{'IsActiveMember':<20} | {feature_importance['IsActiveMember']}\n"
        f"{'Age':<20} | {feature_importance['Age']}\n"
        f"{'Geography_Germany':<20} | {feature_importance['Geography_Germany']}\n"
        f"{'Balance':<20} | {feature_importance['Balance']}\n"
        f"{'Geography_France':<20} | {feature_importance['Geography_France']}\n"
        f"{'Gender_Female':<20} | {feature_importance['Gender_Female']}\n"
        f"{'Geography_Spain':<20} | {feature_importance['Geography_Spain']}\n"
        f"{'CreditScore':<20} | {feature_importance['CreditScore']}\n"
        f"{'EstimatedSalary':<20} | {feature_importance['EstimatedSalary']}\n\n"
        f"Summary statistics for churned customers:\n"
        f"- Avg NumOfProducts: {summary_stats['Avg NumOfProducts']}\n"
        f"- Avg Age: {summary_stats['Avg Age']}\n"
        f"- Avg Balance: {summary_stats['Avg Balance']}\n"
        f"- % IsActiveMember: {summary_stats['IsActiveMember_pct']}%\n"
        f"- % Geography_Germany: {summary_stats['Geography_Germany_pct']}%\n"
        f"- Avg CreditScore: {summary_stats['Avg CreditScore']}\n"
        f"- Avg EstimatedSalary: {summary_stats['Avg EstimatedSalary']}\n\n"
        f"Generate a detailed, customer-friendly explanation (150-200 words) for why {surname} might or might not be at risk of churning. "
        f"Start with: '{intro}' Focus on key features like NumOfProducts, Balance, CreditScore, IsActiveMember, and EstimatedSalary, "
        f"comparing to churned customers. Avoid terms like 'model' or 'probability'."
    )
    
    if model:
        try:
            response = model.generate_content(prompt)
            explanation = response.text.strip()
        except Exception as e:
            print(f"Gemini API error: {e}")
    else:
        print("No Gemini API available. Using fallback logic.")
    
    if 'explanation' not in locals():
        if probability > risk_threshold:
            explanation = (
                f"{intro} Despite a Balance of {processed_input['Balance']}—lower than the {summary_stats['Avg Balance']} average for customers who leave—"
                f"{surname} uses {processed_input['NumOfProducts']} products, more than the typical {summary_stats['Avg NumOfProducts']} for those who depart. "
                f"With an Estimated Salary of {processed_input['EstimatedSalary']}, far above the {summary_stats['Avg EstimatedSalary']} of churned customers, "
                f"and a Credit Score of {processed_input['CreditScore']}, there’s a chance {surname} isn’t fully tapping into our services. "
                f"Being {'' if processed_input['IsActiveMember'] else 'less '}active compared to only {summary_stats['IsActiveMember_pct']}% of those who leave "
                f"suggests we could do more to keep {surname} engaged and satisfied."
            )
        else:
            explanation = (
                f"{intro} With a Credit Score of {processed_input['CreditScore']}, higher than the {summary_stats['Avg CreditScore']} average for those who leave, "
                f"{surname} shows careful financial habits. Using {processed_input['NumOfProducts']} products—above the {summary_stats['Avg NumOfProducts']} norm for churned customers—"
                f"and maintaining a Balance of {processed_input['Balance']}, {surname} aligns with those who stay. "
                f"An Estimated Salary of {processed_input['EstimatedSalary']} exceeds the {summary_stats['Avg EstimatedSalary']} of departed customers, "
                f"and being {'' if processed_input['IsActiveMember'] else 'less '}active compared to {summary_stats['IsActiveMember_pct']}% of them "
                f"reinforces that {surname} likely values our services and feels comfortable here."
            )
    
    return explanation

def generate_email(probability, input_dict, explanation, surname):
    """Generate a detailed retention email with bolded incentives and better spacing."""
    prompt = (
        f"You are a manager at HS Bank, focused on keeping customers engaged with tailored offers. "
        f"A customer named {surname} has a {round(probability * 100, 1)}% chance of churning.\n\n"
        f"Customer information:\n{input_dict}\n\n"
        f"Explanation of their situation:\n{explanation}\n\n"
        f"Generate a detailed, friendly email (200-250 words) to {surname}, encouraging them to stay with HS Bank. "
        f"Highlight positive habits (e.g., credit score, activity) and use the explanation to show why we value them. "
        f"Include 5 tailored incentives in bullet points that start with - , based on their information (e.g., product usage, balance). "
        f"Format each incentive heading in bold using <b> tags (e.g., <b>Incentive Name</b>: Description). "
        f"Add double line breaks between paragraphs and a single line break before each bullet point for readability. "
        f"Do not mention churn probability or machine learning. End with a call-to-action and contact info."
    )
    
    if model:
        try:
            response = model.generate_content(prompt)
            email = response.text.strip()
        except Exception as e:
            print(f"Gemini API error: {e}")
    else:
        print("No Gemini API available. Using fallback logic.")
    
    if 'email' not in locals():
        num_products = input_dict.get("NumOfProducts", 0)
        balance = input_dict.get("Balance", 0)
        credit_score = input_dict.get("CreditScore", 0)
        is_active = input_dict.get("IsActiveMember", 0)
        email = (
            f"Subject: A Special Offer Just For You, {surname}!\n\n"
            f"Dear {surname},\n\n"
            f"We truly value your relationship with HS Bank and appreciate you choosing us for your financial needs. "
            f"{explanation.split('.')[0] + '.'} Your impressive Credit Score of {credit_score} and "
            f"{'' if is_active else 'potential for more '}active engagement make you a fantastic customer we’re excited to support.\n\n"
            f"To show our appreciation and help you get even more from HS Bank, here are some exclusive offers tailored just for you:\n"
            f"\n- <b>Increase your Balance</b>: Boost your balance to $70,000+ and enjoy a 0.5% interest rate increase for a year."
            f"\n- <b>Premium Savings Account</b>: Open a new account and get a 0.75% higher rate for 6 months."
            f"\n- <b>Credit Card Rewards Boost</b>: Apply for our Rewards card and earn 2x points on purchases for 3 months."
            f"\n- <b>Personalized Financial Consultation</b>: Schedule a free session with our advisors to meet your goals."
            f"\n- <b>No Fee Investment Account</b>: Start investing with us and we’ll waive fees for the first year.\n\n"
            f"Ready to take advantage of these offers? Contact Sarah Johnson at 555-123-4567 or email customercare@hsbank.com.\n\n"
            f"Sincerely,\nThe HS Bank Team"
        )
    
    return email