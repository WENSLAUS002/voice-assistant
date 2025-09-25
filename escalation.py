import json
from db import get_available_agent, log_escalation

def escalate_to_agent(user_id, query):
    """Escalates the query to a human agent."""
    agent = get_available_agent()
    if agent:
        log_escalation(user_id, query, agent['id'])
        return {
            "status": "Escalated to human agent",
            "id": agent['id'],
            "message": f"Your query has been assigned to Agent {agent['name']}."
        }
    else:
        return {
            "status": "No agents available",
            "message": "All agents are currently busy. Please try again later."
        }

def handle_escalation(user_id, query, nlp_confidence, response):
    """Decides whether to escalate or ask for rephrasing."""
    CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence required to provide an AI response
    
    if nlp_confidence >= CONFIDENCE_THRESHOLD:
        return {"status": "AI Response", "response": response}
    else:
        # Ask the user to rephrase if no agents are available
        agent_status = escalate_to_agent(user_id, query)
        if agent_status['status'] == "No agents available":
            return {
                "status": "Rephrase Request",
                "message": "I didn't quite understand that. Could you rephrase your question?"
            }
        return agent_status

#usage
def main():
    user_id = 123
    query = "I need help with a transaction error."
    nlp_confidence = 0.4  # Example low confidence score
    response = "I'm sorry, but I can't process this request."
    
    escalation_result = handle_escalation(user_id, query, nlp_confidence, response)
    print(json.dumps(escalation_result, indent=4))

if __name__ == "__main__":
    main()
