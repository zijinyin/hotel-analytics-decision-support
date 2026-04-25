def generate_recommendation(cancel_prob, value_label):
    if cancel_prob >= 0.7 and value_label == 1:
        return "High-priority follow-up: high cancellation risk and high-value customer."
    elif cancel_prob >= 0.7 and value_label == 0:
        return "Monitor booking closely: high cancellation risk but lower customer value."
    elif cancel_prob < 0.7 and value_label == 1:
        return "Stable booking, valuable guest: maintain service quality and engagement."
    else:
        return "Low immediate concern: no urgent action needed."