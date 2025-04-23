### Preprocessing Summary

- Records with complaint narrative retained: 66,806
- Dropped columns: tags, company_public_response, consumer_consent_provided, consumer_complaint_narrative, complaint_id, date_received, date_sent_to_company, submitted_via, zipcode, sub_issue, sub_product, consumer_disputed?, company_response_to_consumer
- Cleaned text: Lowercased, special chars/HTML/URLs/emails removed, extra whitespace stripped
- Binary and encoded fields added: `consumer_disputed_binary`, `timely_response_binary`, `company_response_encoded`
- Text length feature added
- State nulls filled with 'Unknown'

---

####  Sample Cleaned Complaint
> xxxx has claimed i owe them for xxxx years despite the proof of payment i sent them canceled check and their ownpaid invoice for they continue to insist i owe them and collection agencies are after me how can i stop this harassment for a bill i already paid four years ago
