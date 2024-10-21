import google.generativeai as genai
import os
import typing_extensions as typing
import json


genai.configure(api_key="AIzaSyBclKK678ubBkpyC4BO0zsua_8lWizXfgs")

class Similarity(typing.TypedDict):
    agreement_score: float

# user_lines = """
# we should only allow immigration from countries with a GDP greater than that of the United States, and all immigrants shold possess at least a doctorate
# """

user_lines = """
we must address the problem of illegal immigration and deliver a system that is secure  productive  orderly  and fair 
 the president calls on congress to pass comprehensive immigration reform that will secure our borders  enhance interior and worksite enforcement  create a temporary worker program  resolve   without animosity and without amnesty   the status of illegal immigrants already here  and promote assimilation into our society 
 all elements of this problem must be addressed together   or none of them will be solved 
"""

candidate_lines = """
expanding legal immigration   deterring illegal immigration america is a nation of immigrants 
 legal immigration pathways while we wait for congress to act  president biden has used his authority to provide orderly  legal immigration pathways for those fleeing violence  and decrease the number of illegal border crossings 
 legislation must secure the border  reform the asylum system  expand legal immigration  and keep families together by supporting a pathway for long term undocumented individuals  improving the work authorization process  and securing the future of the daca program 
 the legal immigration framework was last updated in      and does not reflect the needs of our country in the   st century 
 democrats believe that asylum processing should be efficient and fair  and that those who are determined not to have a legal basis to remain should be quickly removed 
 we believe that international climate finance is an important tool in this fight 
 he has also taken executive actions  including launching expedited immigration court dockets  to more quickly resolve immigration cases for those crossing the southern border and remove those who do not establish a legal basis to remain 
 we dont believe we should add    trillion to the national debt to give more tax breaks skewed to the wealthy and big corporations 
 in president bidens second term  he will push congress to provide the resources and authorities that we need to secure the border 
 democrats believe that quality  affordable health care should be available in every corner of america 
 we need to fund the police  not defund the police 
 fast  efficient immigration decisions immigration officers have to make the right decisions  and they also have to make them quickly and efficiently 
 this program will ensure communities across the country have access to the capital they need to participate in and benefit from a cleaner  more sustainable economy as we slash harmful climate pollution  improve air quality  lower energy costs  and create good paying jobs 
 president biden has led with the conviction that we must secure our border and fix a broken immigration system decades in the making 
 president biden is also working to improve health care for people with disabilities 
 we will work to protect their voting rights and ensure theyre able to participate in the democratic process here at home 
 scheduling appointments makes the process at our border safer and more orderly  and the advance information that is submitted to cbp creates a more efficient and streamlined process for cbp and for individuals 
 as democrats  we believe the united states has an indispensable role to play in solving the climate crisis  and we have an obligation to help other nations carry out this work 
 a robust immigration system with accessible lawful pathways and penalties for illegal immigration alleviates pressure at the border and upholds our values 
 we continue to strengthen vawa  keep guns out of the hands of domestic abusers  and expand housing and legal services for survivors 
 reform the asylum system congress must pass legislation to reform our asylum system modeled after the bipartisan senate deal so that we can quickly identify and provide protection to those who are fleeing persecution and ensure it is not used as an alternative to legal immigration by others 
 we will improve working conditions and support to help make teaching a sustainable and affordable profession 
 republicans are opposed to this tax relief for families that need it 
 we need to act now and act fast to realize the promise of ai and manage its risks to ensure that ai serves the public interest 
 and we are ensuring access to accurate information and legal resources  including by launching reproductiverights 
 in order to achieve this  we need congress to strengthen requirements for valid asylum claims 
 and  we will remove barriers to legal access  combat hate crimes  and counter cyber threats 
 opportunities to turn a good vice president harris and governor walz believe in thedollars 
 it would have made our country safer and made our border more secure  while treating people fairly and humanely and expanding legal immigration  consistent with our values as a nation 
 we are rooting bias out of the home appraisal process  which perpetuates the racial wealth gap by unjustly undervaluing millions of black  and latino owned homes 
 it will also help to fund community based organizations that host clinics to assist with immigration cases 
 citizenship and immigration services  uscis  has cut processing times significantly for those awaiting work authorization and ensured that immigration decisions including naturalization and work permit applications are made fairly and as quickly as possible 
 cut taxes for middle class families vice president harris and governor walz believe that working families deserve a break 
 women in the workforce need family focused policy 
 and  we will strengthen legal protections for and support survivors of deepfake image based sexual abuse building on the federal civil cause of action established under the presidents reauthorization of vawa in      
 he has sought increased funding to improve the prevention  reporting and prosecution of hate crimes 
 entrepreneurs need to have access to customers to start or expand their business 
 fighting poverty as democrats  we believe in an america where people look out for one another and leave no one behind 
 we will improve and speed up the processes of environmental review and clean energy permitting  and further scale up development of clean energy on public lands 
 we will also improve and increase access to mental health care  expand suicide prevention  and invest in opioid overdose prevention and treatment 
 and  president biden reestablished the justice departments standalone office for access to justice to ensure that everyone can get legal assistance 
 democrats support funding to improve critical health care facilities in the territories  including construction of a state of the art medical complex in guam that co locates health agencies on a single campus for a more integrated healthcare system that will benefit the region 
 education democrats fundamentally believe that every student deserves a quality education  regardless of their parents zip code or income 
 as commander in chief  she will ensure that the united states military remains the strongest  most lethal fighting force in the world  that we unleash the power of american innovation and win the competition for the   st century  and that we strengthen  not abdicate  our global leadership 
 well also keep pushing to improve pay and benefits for care workers  for example by fighting to get many of them a bigger share of medicaid home care payments 
 they believe no child in america should livein poverty  and this reform will have a historic impact 
  way forward that not only protects these achievements but knowing that they can afford the medications they need 
 we will establish a national  comprehensive paid family and medical leave program to ensure that all workers  including women  can take the time they need to bond with a new child  care for a loved one  or recover from an illness 
 they believe no child in america should live in poverty  and these actions would have a historic impact 
 and he has supported our regional partners through the la declaration process working together on coordinated enforcement  development of lawful pathways  and efforts to address the root causes of migration 
 """

model = genai.GenerativeModel("gemini-1.5-flash")
response = model.generate_content(
    f"""
    I will provide two inputs: one titled user_lines and one titled candidate_lines. 
    user_lines contains a few lines containing a user's position in a certain policy domain. 
    candidate_lines contains 50 lines containing a candidate's position in the same policy domain. 
    Create an output that is a value between -1 to 1 that represents how closely you believe the user's
      position matches with the candidate's postion. You are allowed and encouranged to output a decimal value, 
      and the general set of all opinions fed into this model is approximately Gaussian centered at 0.
      -1 represents diametrically opposed, 
      while 1 represents perfectly aligned, 
      and 0 means that there is no conflict nor agreement.
      Please output a value with up to two decimals of precision.

    user_lines: {user_lines}

    candidate_lines: {candidate_lines}
    """,
    generation_config=genai.GenerationConfig(
        response_mime_type="application/json", response_schema=list[Similarity]
    ),
)

result = response.text

# extract the float from the result string
agreement = json.loads(result)[0]['agreement_score']
print(agreement)