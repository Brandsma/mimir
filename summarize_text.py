from util.logger import setup_logger
from transformers import pipeline

# Alt models
# "pszemraj/bigbird-pegasus-large-K-booksum" -> https://huggingface.co/pszemraj/bigbird-pegasus-large-K-booksum
# gogamza/kobart-summarization"

log = setup_logger(__name__)


def summarize_text(text):
    assert type(text) == str, "text is not a string"

    # Make model
    summary_generator = pipeline("summarization", model="pszemraj/bigbird-pegasus-large-K-booksum") 
    # Generate summary
    summary = summary_generator("summarize: " + text)[0]['summary_text']

    return summary

def one_line_summary(text):
    assert type(text) == str, "text is not a string"

    # Make model
    one_line_summary_generator = pipeline("summarization", model="snrspeaks/t5-one-line-summary")
    # Generate summary
    one_line_summary = one_line_summary_generator("summarize: " + text)[0]['summary_text']
    print(one_line_summary)
    return one_line_summary


if __name__ == "__main__":
    context = """Max Verstappen will be wearing Red Bull colours for the foreseeable after signing a huge five-year contract extension that will keep him at the four-time constructors’ champions until at least the end of 2028. But with two years remaining on the Dutchman’s current deal, why extend now? 
                Verstappen struck while the iron was hot 
                The Verstappen family are shrewd operators, and after giving Red Bull their first world title since 2013, they were in a strong position to negotiate a long-term arrangement on considerably better terms. 
                Red Bull have made no secret of their support for Verstappen, effectively moulding the team around him, as they feel that’s the best way to not just secure world titles but deliver a sustained period of success.
                Verstappen delivered last year with a title win – and he will know his newfound worth. He’s hot property right now. With Lewis Hamilton coming towards the end of his career, a berth at Mercedes at some point was feasible. Ferrari and McLaren could be options, too.
                Any team on the grid would happily recruit Verstappen – and when you’re in that kind of demand, you can afford to squeeze a team who you know need and want you.
                So despite still having two years left, Verstappen brought Red Bull to the table, and the result is a new five-year deal – which sources say is on much improved terms and was agreed in principle in Barcelona last week.
                Verstappen gave Red Bull their first drivers' crown since 2013 last season
                The deal is the longest of any driver on the grid, and it makes a lot of sense for the 24-year-old. Verstappen feels at home at Red Bull. It’s where he feels he belongs. He loves the team and those he works with – and he’s got an incredible relationship with senior management.
                The decision was quite straightforward, both sides wanted to continue, said Verstappen. From the start, I’ve felt really good in the team, especially last year. I only saw one way forward and that was with this team.
                Staying put, especially when the team have shown they can deliver a world championship-winning car and while they’re building their own power unit division with an eye on long-term success, was a no-brainer.
                I just feel really good at the team and I really enjoy working with the people in every department, especially of course after winning the championship last year, he added. For me, it’s the best team out there so I want to stay.
                Red Bull need Verstappen to secure future
                Red Bull exist to win, so the eight years before Verstappen’s title victory last year were painful. They’ve heavily invested in the Dutchman, and took a risk on giving him his F1 debut aged 17 – so they want to reap the rewards.
                After he clinched the title, they knew they would have to up their game to keep him on, not least because he’s proving he’s a once-in-a-generational talent.
                Verstappen has shown loyalty to Red Bull, after they brought him into F1 aged 17
                With significant sponsor deals over the winter – including Oracle, whose deal is worth around $500m, and Bybit, who are pumping in around $150m – as well as the continued support from Red Bull owner Dietrich Mateschitz, they had the funds to lock in Verstappen for the long-term, ensuring crucial stability.
                With Sergio Perez at the tail of his career, Red Bull are limited with their options for leading lights. Lando Norris has signed a bumper deal with McLaren, which keeps him at the team until the end of 2025, while Charles Leclerc is set to stay at Ferrari until at least the conclusion of 2024. And George Russell is a long-term driver at Mercedes. So Red Bull had to move to keep Verstappen in the family.
                I think it very much demonstrates the commitment that Red Bull has to Max and Max has to Red Bull, said Horner. This is fantastic news for the whole team, it shows commitment from both sides and a real belief in what we’re doing. So what better way to start the new season with the extension of this agreement until the end of the 2028 season. It’s phenomenal for us
                Play Video
                Max Verstappen - The Rise of a Champion
                And Verstappen’s commitment, which will run through the engine formula overhaul for 2026, when Red Bull Powertrains are set to debut their first in -house power unit, makes a lot of sense.
                I think it was very important for us to have continuity, added Horner. We’ve always been a big believer in continuity. I think having Max for the long-term, through the transition into 2026 with the new regulations, as we become an engine manufacturer and supplier, is fantastic
                It may have cost them – but Verstappen will be worth every penny to Red Bull."""

    abstract = """We describe a system called Overton, whose main design goal is to support engineers in building, monitoring, and improving production 
                machine learning systems. Key challenges engineers face are monitoring fine-grained quality, diagnosing errors in sophisticated applications, and 
                handling contradictory or incomplete supervision data. Overton automates the life cycle of model construction, deployment, and monitoring by providing a 
                set of novel high-level, declarative abstractions. Overton's vision is to shift developers to these higher-level tasks instead of lower-level machine learning tasks. 
                In fact, using Overton, engineers can build deep-learning-based applications without writing any code in frameworks like TensorFlow. For over a year, 
                Overton has been used in production to support multiple applications in both near-real-time applications and back-of-house processing. In that time, 
                Overton-based applications have answered billions of queries in multiple languages and processed trillions of records reducing errors 1.7-2.9 times versus production systems.
                """

    log.info(summarize_text(context))
