class RecommendationModel:

    def __init__(self,payment_amount,reff_reason):
        self.payment_amount = payment_amount
        self.reff_reason = reff_reason
    
    def __str__(self):
        return f"payment_amount : {self.payment_amount} , reff_reason :{self.reff_reason} "
