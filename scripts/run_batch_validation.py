import json
import time
import requests
import os

SERVER_URL = "http://localhost:8000/generate"
REPORT_FILE = "results/validation_report.json"

# --- Golden Set Data ---
GOLDEN_SET = [
    {"id": 1, "question": "What are the detailed requirements for implementing account management processes in AC-2, including automated mechanisms for disabling inactive accounts?", "keywords": ["account management", "automated mechanisms", "disable inactive", "90 days"]},
    {"id": 2, "question": "Explain the specific controls for managing access authorizations in AC-3, focusing on the enforcement of approved authorizations for logical access.", "keywords": ["access enforcement", "approved authorizations", "logical access", "information flow"]},
    {"id": 3, "question": "Describe the technical implementations required for information flow enforcement in AC-4, including the use of security policy filters.", "keywords": ["information flow", "security policy", "filters", "enforce approved"]},
    {"id": 4, "question": "What are the separation of duties requirements outlined in AC-5, and how do they integrate with access control mechanisms?", "keywords": ["separation of duties", "access control", "conflicting duties", "least privilege"]},
    {"id": 5, "question": "Detail the least privilege principles in AC-6, including the explicit prohibition of unnecessary privileges for users and processes.", "keywords": ["least privilege", "explicit prohibit", "unnecessary privileges", "users processes"]},
    {"id": 6, "question": "What specific measures are required for unsuccessful logon attempts in AC-7, including lockout durations and automated responses?", "keywords": ["unsuccessful logon", "lockout duration", "automated response", "threshold"]},
    {"id": 7, "question": "Explain the system use notification requirements in AC-8, including the content and display of warning banners.", "keywords": ["system use notification", "warning banners", "display before", "authorized use"]},
    {"id": 8, "question": "Describe the monitoring and control of concurrent sessions in AC-10, with emphasis on limiting sessions per user.", "keywords": ["concurrent sessions", "limit sessions", "per user", "monitoring control"]},
    {"id": 9, "question": "What are the detailed requirements for session lock mechanisms in AC-11, including pattern-hiding displays?", "keywords": ["session lock", "pattern-hiding", "inactivity period", "retain until"]},
    {"id": 10, "question": "Outline the wireless access restrictions in AC-18, focusing on authentication and encryption protocols.", "keywords": ["wireless access", "authentication encryption", "monitor control", "unauthorized use"]},
    {"id": 11, "question": "What specific audit events must be recorded under AU-2, including the rationale for selecting those events?", "keywords": ["audit events", "organization-defined", "successful unsuccessful", "risk assessment"]},
    {"id": 12, "question": "Detail the content requirements for audit records in AU-3, such as timestamps and event types.", "keywords": ["audit records", "timestamps", "event types", "user identities"]},
    {"id": 13, "question": "Explain the audit storage capacity management in AU-4, including allocation and overflow handling.", "keywords": ["audit storage", "capacity allocation", "overflow handling", "prevent loss"]},
    {"id": 14, "question": "Describe the response actions for audit processing failures in AU-5, like alerting personnel.", "keywords": ["audit processing failures", "alert personnel", "automated actions", "shutdown overwrite"]},
    {"id": 15, "question": "What are the monitoring requirements for audit log reviews in AU-6, including frequency and correlation with other data?", "keywords": ["audit log review", "frequency", "correlation analysis", "real-time alerts"]},
    {"id": 16, "question": "Outline the audit record reduction and report generation processes in AU-7, ensuring non-repudiation.", "keywords": ["audit reduction", "report generation", "non-repudiation", "original records"]},
    {"id": 17, "question": "Detail the time stamp synchronization in AU-8 using authoritative sources.", "keywords": ["time stamps", "authoritative source", "synchronization", "internal clocks"]},
    {"id": 18, "question": "Explain the protection measures for audit information in AU-9, including access controls.", "keywords": ["protect audit information", "access controls", "hardware software", "cryptographic"]},
    {"id": 19, "question": "What are the non-repudiation assurances provided by AU-10 using digital signatures?", "keywords": ["non-repudiation", "digital signatures", "audit records", "integrity assurance"]},
    {"id": 20, "question": "Describe the audit record retention periods and methods in AU-11.", "keywords": ["audit record retention", "organization-defined period", "legal requirements", "dispose securely"]},
    {"id": 21, "question": "What specific boundary protection mechanisms are required in SC-7, such as firewalls and gateways?", "keywords": ["boundary protection", "firewalls gateways", "monitor control", "external interfaces"]},
    {"id": 22, "question": "Explain the transmission confidentiality and integrity protections in SC-8 using cryptography.", "keywords": ["transmission confidentiality", "integrity protection", "cryptographic", "alternative safeguards"]},
    {"id": 23, "question": "Detail the network disconnect requirements in SC-10, including timeout periods.", "keywords": ["network disconnect", "timeout periods", "idle sessions", "terminate connections"]},
    {"id": 24, "question": "Describe the cryptographic key establishment and management in SC-12.", "keywords": ["cryptographic key", "establishment management", "secure generation", "distribution storage"]},
    {"id": 25, "question": "What are the requirements for using FIPS-validated cryptographic modules in SC-13?", "keywords": ["FIPS-validated", "cryptographic modules", "confidentiality integrity", "security strength"]},
    {"id": 26, "question": "Outline the public key infrastructure certificate handling in SC-17.", "keywords": ["public key infrastructure", "certificates", "issue manage", "validation revocation"]},
    {"id": 27, "question": "Explain the mobile code controls in SC-18, including usage restrictions.", "keywords": ["mobile code", "usage restrictions", "risk assessment", "authorize monitor"]},
    {"id": 28, "question": "Detail the voice over IP protections in SC-19.", "keywords": ["voice over IP", "protections", "confidentiality integrity", "signaling media"]},
    {"id": 29, "question": "What secure name/address resolution services are required in SC-20?", "keywords": ["secure name resolution", "address resolution", "authoritative source", "data integrity"]},
    {"id": 30, "question": "Describe the session authenticity measures in SC-23 using cryptographic techniques.", "keywords": ["session authenticity", "cryptographic techniques", "protect authenticity", "invalidate keys"]},
    {"id": 31, "question": "What flaw remediation processes are outlined in SI-2, including scanning and patching?", "keywords": ["flaw remediation", "scanning patching", "identify report", "correct flaws"]},
    {"id": 32, "question": "Explain the malicious code protection mechanisms in SI-3, such as antivirus updates.", "keywords": ["malicious code protection", "antivirus updates", "scan detect", "block quarantine"]},
    {"id": 33, "question": "Detail the system monitoring requirements in SI-4, including intrusion detection tools.", "keywords": ["system monitoring", "intrusion detection", "tools sensors", "alert analyze"]},
    {"id": 34, "question": "Describe the security alerts, advisories, and directives handling in SI-5.", "keywords": ["security alerts", "advisories directives", "receive disseminate", "implement actions"]},
    {"id": 35, "question": "What are the integrity checking mechanisms for software and firmware in SI-7?", "keywords": ["integrity checking", "software firmware", "detect unauthorized", "baseline comparison"]},
    {"id": 36, "question": "Outline the spam protection measures in SI-8.", "keywords": ["spam protection", "detect prevent", "email web", "update signatures"]},
    {"id": 37, "question": "Explain the input validation controls in SI-10 to prevent invalid inputs.", "keywords": ["input validation", "prevent invalid", "check validity", "reject sanitize"]},
    {"id": 38, "question": "Detail the error handling requirements in SI-11 to avoid information disclosure.", "keywords": ["error handling", "information disclosure", "generate messages", "avoid sensitive"]},
    {"id": 39, "question": "What memory protection techniques are required in SI-16, such as address space layout randomization?", "keywords": ["memory protection", "address space", "layout randomization", "prevent exploitation"]},
    {"id": 40, "question": "Describe the configuration management policy and procedures in CM-1.", "keywords": ["configuration management", "policy procedures", "develop document", "disseminate review"]},
    {"id": 41, "question": "Explain the baseline configuration maintenance in CM-2, including updates and approvals.", "keywords": ["baseline configuration", "maintain update", "approvals", "as-built documentation"]},
    {"id": 42, "question": "Detail the change control processes in CM-3, focusing on testing and validation.", "keywords": ["change control", "testing validation", "approve reject", "document changes"]},
    {"id": 43, "question": "What security impact analyses are required in CM-4 for changes?", "keywords": ["security impact analysis", "system changes", "assess impact", "before implementation"]},
    {"id": 44, "question": "Outline the access restrictions for configuration changes in CM-5.", "keywords": ["access restrictions", "configuration changes", "enforce authorize", "audit logging"]},
    {"id": 45, "question": "Describe the cryptography management in CM-12 for configuration items.", "keywords": ["cryptography management", "configuration items", "protect confidentiality", "key management"]},
    {"id": 46, "question": "Explain the incident response policy and procedures in IR-1.", "keywords": ["incident response", "policy procedures", "develop distribute", "review update"]},
    {"id": 47, "question": "Detail the incident response training requirements in IR-2, including simulated events.", "keywords": ["incident response training", "simulated events", "provide conduct", "test capabilities"]},
    {"id": 48, "question": "What incident response testing is outlined in IR-3, such as tabletop exercises?", "keywords": ["incident response testing", "tabletop exercises", "conduct test", "evaluate effectiveness"]},
    {"id": 49, "question": "Describe the incident monitoring tools and techniques in IR-4.", "keywords": ["incident monitoring", "tools techniques", "detect report", "automated mechanisms"]},
    {"id": 50, "question": "Outline the incident reporting timelines and recipients in IR-6.", "keywords": ["incident reporting", "timelines recipients", "report to", "internal external"]}
]

def main():
    results = []
    success_count = 0
    total_latency = 0
    
    print(f"Starting Batch Validation against {SERVER_URL}...")
    
    for item in GOLDEN_SET:
        item_id = item['id']
        question = item['question']
        expected_keywords = item['keywords']
        
        try:
            start = time.time()
            response = requests.post(SERVER_URL, json={"text": question}, timeout=60)
            latency = time.time() - start
            
            if response.status_code == 200:
                data = response.json()
                generated = data.get("generated_text", "")
                
                # Check keywords
                found_keywords = [k for k in expected_keywords if k.lower() in generated.lower()]
                is_valid = len(found_keywords) > 0
                
                if is_valid:
                    success_count += 1
                
                results.append({
                    "id": item_id,
                    "status": "success",
                    "latency_seconds": round(latency, 4),
                    "keywords_found": found_keywords,
                    "keywords_expected": expected_keywords
                })
            else:
                results.append({
                    "id": item_id,
                    "status": f"error_{response.status_code}",
                    "error": response.text
                })
                
        except Exception as e:
            results.append({
                "id": item_id,
                "status": "exception",
                "error": str(e)
            })
            
    # Summary
    total_items = len(GOLDEN_SET)
    success_rate = (success_count / total_items) * 100
    avg_latency = sum(r.get('latency_seconds', 0) for r in results) / total_items
    
    report = {
        "summary": {
            "total_items": total_items,
            "success_count": success_count,
            "success_rate_percentage": round(success_rate, 2),
            "average_latency_seconds": round(avg_latency, 4)
        },
        "details": results
    }
    
    # Save Report
    with open(REPORT_FILE, 'w') as f:
        json.dump(report, f, indent=2)
        
    print(f"\n--- Validation Complete ---")
    print(f"Success Rate: {success_rate:.2f}%")
    print(f"Avg Latency:  {avg_latency:.4f}s")
    print(f"Report saved to {REPORT_FILE}")

if __name__ == "__main__":
    main()
