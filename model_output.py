import os
from typing import Dict
from transformers import BertForSequenceClassification, BertTokenizer, GPT2LMHeadModel, GPT2Tokenizer, T5ForConditionalGeneration, T5Tokenizer
from sentence_transformers import SentenceTransformer
import torch
import numpy as np

# BERT label mapping (from custom_bert_component.py)
LABELS = {
    'Account Balance': 0, 'Password Reset': 1, 'International Transfer': 2, 'Working Hours': 3,
    'Opening Savings Account': 4, 'Finding Routing/Account Numbers': 5, 'Routing Number Usage': 6,
    'ABA and Routing Number Definition': 7, 'List of Routing Numbers': 8, 'Accessing Account': 9,
    'Finding Account Number': 10, 'Finding Financial Center': 11, 'Changing Address': 12,
    'Changing Legal Name/Title': 13, 'Closing Account': 14, 'Stopping Payment on Check': 15,
    'Withdrawing Money': 16, 'Depositing Money': 17, 'Deposit Hold Information': 18,
    'After-Hours Transactions': 19, "Cashier's Check": 20, "Cashier's Check Alternatives": 21,
    'Scheduled Transfer': 22, 'Bank Holiday Schedule': 23, 'FDIC Definition': 24,
    'FDIC Insured Accounts': 25, 'FDIC Insurance Limits': 26, 'FDIC Transaction Account Guarantee': 27,
    'FDIC Insurance Coverage': 28, 'Abandoned Property Notification': 29, 'Online Account Statements': 30,
    'Online Account Services': 31, 'Reorder Checks': 32, 'Update Contact Information': 33,
    'Contact Bank of America': 34, 'Open an Account': 35, 'Application Requirements': 36,
    'Joint Account Information': 37, 'Initial Deposit': 38, 'Deposit Amount': 39,
    'Application Security': 40, 'Account Opening Timeline': 41, 'Offer Code': 42,
    'Online Account Access': 43, 'Account Opening Documents': 44, 'Signature Form': 45,
    'Save Application': 46, 'Nonpermanent Resident Account': 47, 'Order ATM Card': 48,
    'Replace ATM Card': 49, 'Activate ATM Card': 50, 'Lock/Unlock ATM Card': 51,
    'Fastest ATM Card Delivery': 52, 'Change/Request PIN': 53, 'Non-Working PIN': 54,
    'Dispute Transaction': 55, 'Lost/Stolen Card': 56, 'Lock ATM Card': 57,
    'Replacement Card Fees': 58, 'ATM Usage Fees': 59, 'ATM Transactions': 60,
    'ATM Fees': 61, 'ADA Compliance': 62, 'Max Withdrawal': 63, 'Adjust Withdrawal Limit': 64,
    'Cash Preference': 65, 'Denominations': 66, 'Same-Day Deposits': 67, 'nan': 68,
    'Next-Day Deposits': 69, 'Funds Availability': 70, 'Foreign Deposits': 71,
    'Split Deposits': 72, 'Coin Deposits': 73, 'Receipt Options': 74, 'Check Images': 75,
    'ATM Transfers': 76, 'ATM Payments': 77, 'Email Receipts': 78, 'ATM Preferences': 79,
    'PIN Change': 80, 'Email Validation': 81, 'Email Management': 82, 'ATM Receipt Viewing': 83,
    'Contactless ATMs': 84, 'Contactless Card Use': 85, 'Digital Wallet Use': 86,
    'Stolen Device Security': 87, 'ATM Maintenance Reporting': 88, 'Incorrect Transaction Claim': 89,
    'Deposit Discrepancy': 90, 'Missing Receipt': 91, 'Card Retention': 92,
    'Foreign Cash Dispensing': 93, 'Global ATM Alliance': 94, 'Foreign ATM Tips': 95,
    'International ATM Issues': 96, 'Foreign Currency Preparation': 97, 'Non-Customer ATM Use': 98,
    'ATM Card Security': 99, 'ATM Surroundings Safety': 100, 'ATM Privacy Protection': 101,
    'Emergency Assistance': 102, 'Overdraft Item Fee': 103, 'Overdraft Protection Transfer Fee': 104,
    'Daily Overdraft Fee Limit': 105, 'Safe Deposit Box Cost': 106, 'Checking Account Fees': 107,
    'Monthly Maintenance Fees': 108, 'Combined Balances for Fee Waiver': 109,
    'Qualifying Direct Deposit': 110, 'Savings Account Cost': 111, 'CD Rates': 112,
    'Check Image Service': 113, 'Stop Payment Fee': 114, 'Late Fees': 115,
    'Non-Bank of America ATM Fee': 116, 'Benefits of Linking Accounts': 117,
    'Automatic Account Linking': 118, 'Requesting Account Linking': 119,
    'Information on Linking Accounts': 120, 'Paying Bills Without Checks': 121,
    'Direct Deposit Without Voided Check': 122, 'POD Beneficiary Benefits': 123,
    'Eligible POD Accounts': 124, 'Multiple POD Beneficiaries': 125,
    'POD Beneficiaries Predecease Owner': 126, "Overdrawn Account at Owner's Death": 127,
    'Eligible POD Beneficiaries': 128, 'POD for Joint Account Owners': 129,
    'Why Owners Cannot Be POD Beneficiaries': 130, 'Owner as POD Beneficiary': 131,
    'Beneficiary Cannot Be Added Error': 132, 'Review Beneficiary Error': 133,
    'POD Save Error': 134, 'Online Bill Pay Setup': 135, 'Bill Pay Overview': 136,
    'eBills Explanation': 137, 'Pay To or Pay From Accounts': 138, 'Bill Pay Limit': 139,
    'Payment Processing Time': 140, 'Bill Pay Fee': 141, 'eBill Functionality': 142,
    'Cancel eBill Process': 143, 'Pay eBill Process': 144, 'View eBills Location': 145,
    'Bill Payment Subtraction Timing': 146, 'CD Minimum Opening Balance': 147,
    'CD Maximum Online Opening Balance': 148, 'CD Rates Location': 149,
    'CD Maturity Grace Period': 150, 'CD maturity Grace Period': 151,
    'CD Maturity Notification': 152, 'CD Renewal Process': 153,
    'CD Application Processing Time': 154, 'CD Interest Accrual Start': 155,
    'CD Interest Rate Application': 156, 'CD Early Withdrawal Penalty': 157,
    'IRA Definition': 158, 'Roth IRA Definition': 159, 'Traditional IRA Definition': 160,
    'Roth vs Traditional IRA Differences': 161, 'IRA Contribution Limits': 162,
    'IRA with Employer Plans': 163, 'Rollover IRA Definition': 164, 'Rollover vs Transfer': 165,
    'Direct vs Indirect Rollover': 166, 'Rollover IRA Transfer': 167,
    'Post-Marriage Account Update': 168, 'Post-Divorce Account Update': 169,
    'Power of Attorney Importance': 170, 'Debit Card Ordering': 171,
    'Debit Card Replacement': 172, 'Debit Card Activation': 173, 'Debit Card Locking': 174,
    'Debit Card Expedited Delivery': 175, 'ATM to Debit Card Replacement': 176,
    'Debit Card Alerts Setup': 177, 'Debit Card ATM Limits': 178,
    'Replacement Debit Card Fees': 179, 'Debit Card Transaction Fees': 180,
    'Chip Card Fees': 181, 'Non-Bank Teller Withdrawal Fees': 182, 'Lost or Stolen Card': 183,
    'Lock Debit Card': 184, 'Change or Request PIN': 185, 'Sign vs. PIN Authorization': 186,
    'Digital Debit Card Definition': 187, 'Obtain Digital Debit Card': 188,
    'PIN for Digital Debit Card': 189, 'Expiration Date of Digital Debit Card': 190,
    'Add to Digital Wallet': 191, 'Manage Digital Debit Card': 192, 'CVV Change': 193,
    'ATM Use': 194, 'Digital Wallet Definition': 195,
    'A virtual card for your personal account is a digital substitute for your physical debit card, featuring a unique card number distinct from your physical card.': 196,
    'Virtual Card Definition': 197, 'Virtual vs. Digital Card': 198, 'Authorization Definition': 199,
    'Pending vs. Final Amount': 200, 'Final Amount Posting': 201, 'Pending Transaction Errors': 202,
    'Debit Card ATM Services': 203, 'Debit vs. Credit Card': 204, 'Linking Accounts': 205,
    'Affinity Debit Card': 206, 'Chip Card': 207, 'Removing Hold': 208, 'Avoiding Hold': 209,
    'Accessing Funds': 210, 'Reason for Hold': 211, 'Hold Notification': 212,
    'Hold Duration': 213, 'Check Processing': 214, 'Third-Party Verification': 215,
    'Direct Deposit Function': 216, 'Direct Deposit Setup': 217, 'Prefilled Form Access': 218,
    'Routing/Account Numbers': 219, 'Deposit Alerts': 220, 'Direct Deposit Cost': 221,
    'Voided Check Requirement': 222, 'ABA Routing Number Location': 223,
    'Checking Account Number Location': 224, 'Social Security Direct Deposit Setup': 225,
    'Federal Benefits Direct Deposit Enrollment': 226, 'My Financial Picture Usage': 227,
    'External Account Definition': 228, 'Non-Financial Account Linking': 229,
    'Missing Transactions': 230, 'External Account Error Codes': 231,
    'Downloading My Financial Picture Data': 232, 'Keep the Change Overview': 233,
    'Keep the Change Enrollment': 234, 'Keep the Change Round-Up Transfers': 235,
    'Tracking Keep the Change Savings': 236, 'Stopping Keep the Change Program': 237,
    'Mobile Check Deposit Checks': 238, 'Fund Availability': 239, 'Deposit Issues': 240,
    'Monthly Deposit Limits': 241, 'Cutoff Time for Next-Day Availability': 242, 'Fees': 243,
    'Troubleshooting Tips': 244, 'Taking a Good Picture': 245, 'Terms and Conditions': 246,
    'Features and Benefits': 247, 'Enrollment Process': 248, 'Cost': 249,
    'Checking Without Enrollment': 250, 'Accessing Account Information': 251,
    'Ordering Process': 252, 'Cost of Checks': 253, 'Order Status': 254, 'Functionality': 255,
    'Eligible Accounts': 256, 'Setup or Changes': 257, 'Overdraft Fees': 258,
    'Multiple Accounts': 259, 'Overdraft Definition': 260, 'Available Balance': 261,
    'Insufficient Funds': 262, 'Avoid Overdrafts': 263, 'Transaction Order': 264,
    'Overdraft Fee Reason': 265, 'Avoid Overdraft Transactions': 266, 'Balance Connect': 267,
    'Open Safe Deposit Box': 268, 'Change Safe Deposit Box': 269, 'Savings Account Types': 270,
    'Minimum Balance': 271, 'Interest Rates': 272, 'Apply for Savings': 273,
    'Interest Accrual': 274, 'Interest Payment': 275, 'Keep the Change': 276,
    'Online Security': 277, 'Online Banking Requirements': 278, 'International Access': 279,
    'View Statements': 280, 'Statement Availability': 281, 'Go Paperless': 282,
    'Paper Statement Delivery': 283, 'Paperless Fee': 284, 'Paperless Benefits': 285,
    'Balance Worksheet': 286, 'Online Statement Copy': 287, 'Paper Statement Copy': 288,
    'Statement Retention': 289, 'View Check Images': 290, 'Check Images Fee': 291,
    'Paper Check Copy': 292, 'Combined Statement Images': 293, 'Check Image Retention': 294,
    'Proof of Payment': 295, 'Check 21': 296, 'Braille/Large-Print Images': 297,
    'Student Accounts': 298, 'Child College Account': 299, 'Tax Forms': 300,
    'View/Print Statements': 301, 'Tax-Deductible Transactions': 302, 'Older Statements': 303,
    'Canceled Check Images': 304, 'Order Canceled Check': 305, 'Find Transaction': 306,
    'Tax Refund': 307, 'ABA Routing Number': 308, 'Tax Deadline': 309, 'Zelle Transfer': 310,
    'Other Transfers': 311, 'Recurring Transfers': 312, 'Credit Card Transfer': 313,
    'Mortgage/Loan Transfer': 314, 'Business Account Transfer': 315,
    'Business-Personal Transfer': 316, 'Wire Transfer Requirements': 317,
    'Schedule Appointment': 318, 'Appointment Benefits': 319, 'Chat Session': 320,
    'Verify Secure Chat': 321, 'Chat Products No Login': 322, 'Chat Invitation Timing': 323,
    'Prevent Chat Invitation': 324, 'Print or Email Chat': 325, 'Contact via Email': 326,
    'Contact via Phone or Mail': 327, 'Find Financial Center': 328,
    'Wire Transfer Definition': 329, 'Remittance Transfer Definition': 330,
    'Send Wire Transfer': 331, 'Fee Error Handling': 332, 'USD-Only Countries': 333,
    'Business Wire Fees': 334, 'Assert Remittance Error': 335, 'Cancel Remittance Transfer': 336,
    'Receive Wire Transfer': 337, 'Wire Transfer Fees and Limits': 338,
    'Send Wire Requirements': 339, 'SWIFT Code': 340, 'IBAN': 341,
    'Bank of America IBAN': 342, 'International Wire Timing': 343
}

# Reverse label mapping for BERT
REVERSE_LABELS = {v: k for k, v in LABELS.items()}
REVERSE_LABELS[68] = "fallback"

def test_bert(model_path: str, input_text: str) -> None:
    """Test BERT model output."""
    if not os.path.exists(model_path):
        print(f"BERT model directory '{model_path}' does not exist.")
        return
    try:
        model = BertForSequenceClassification.from_pretrained(model_path)
        tokenizer = BertTokenizer.from_pretrained(model_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        intent_id = logits.argmax().item()
        intent = REVERSE_LABELS.get(intent_id, "Unknown")
        confidence = float(torch.softmax(logits, dim=1)[0, intent_id])
        print(f"BERT prediction: Intent='{intent}', Confidence={confidence:.4f}")
    except Exception as e:
        print(f"BERT error: {str(e)}")

def test_gpt2(model_path: str, input_text: str) -> None:
    """Test GPT-2 model output."""
    if not os.path.exists(model_path):
        print(f"GPT-2 model directory '{model_path}' does not exist.")
        return
    try:
        model = GPT2LMHeadModel.from_pretrained(model_path)
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        inputs = tokenizer(input_text, return_tensors="pt", padding=True)
        outputs = model.generate(
            **inputs,
            max_length=50,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"GPT-2 output: {decoded}")
    except Exception as e:
        print(f"GPT-2 error: {str(e)}")

def test_sbert(model_path: str, sentences: list) -> None:
    """Test SBERT model output."""
    if not os.path.exists(model_path):
        print(f"SBERT model directory '{model_path}' does not exist.")
        return
    try:
        model = SentenceTransformer(model_path)
        embeddings = model.encode(sentences)
        print(f"SBERT embeddings: Shape={embeddings.shape}")
    except Exception as e:
        print(f"SBERT error: {str(e)}")

def test_t5(model_path: str, input_text: str) -> None:
    """Test T5 model output."""
    if not os.path.exists(model_path):
        print(f"T5 model directory '{model_path}' does not exist.")
        return
    try:
        model = T5ForConditionalGeneration.from_pretrained(model_path)
        tokenizer = T5Tokenizer.from_pretrained(model_path)
        inputs = tokenizer(input_text, return_tensors="pt", padding=True)
        outputs = model.generate(
            **inputs,
            max_length=50,
            num_return_sequences=1,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"T5 output: {decoded}")
    except Exception as e:
        print(f"T5 error: {str(e)}")

if __name__ == "__main__":
    input_text = "What's my balance?"
    sbert_sentences = [input_text, "Your balance is $1000."]

    print("Testing BERT...")
    test_bert("./models/bert_model", input_text)

    print("\nTesting GPT-2...")
    test_gpt2("./models/gpt_model", input_text)

    print("\nTesting SBERT...")
    test_sbert("./models/sbert_model", sbert_sentences)

    print("\nTesting T5...")
    test_t5("./models/t5_model", f"question: {input_text}")