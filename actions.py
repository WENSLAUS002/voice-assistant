from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.forms import FormValidationAction
from rasa_sdk.types import DomainDict

class ValidateTransferForm(FormValidationAction):
    def name(self) -> Text:
        return "validate_transfer_form"

    def validate_amount(self, slot_value: Any, dispatcher: CollectingDispatcher, tracker: Tracker, domain: DomainDict) -> Dict[Text, Any]:
        if isinstance(slot_value, (int, float)) and slot_value > 0:
            return {"amount": slot_value}
        dispatcher.utter_message(text="Please provide a valid amount greater than 0.")
        return {"amount": None}

    def validate_recipient(self, slot_value: Any, dispatcher: CollectingDispatcher, tracker: Tracker, domain: DomainDict) -> Dict[Text, Any]:
        if slot_value and isinstance(slot_value, str):
            return {"recipient": slot_value}
        dispatcher.utter_message(text="Please provide a valid recipient name.")
        return {"recipient": None}

    def validate_date(self, slot_value: Any, dispatcher: CollectingDispatcher, tracker: Tracker, domain: DomainDict) -> Dict[Text, Any]:
        if slot_value and isinstance(slot_value, str):
            return {"date": slot_value}
        dispatcher.utter_message(text="Please provide a valid date.")
        return {"date": None}