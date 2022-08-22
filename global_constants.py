import os
# from collections import defaultdict


class CONST:
    """
    constants that does not change during the
            *****PROJECT*****
    for execution settings, see specific scripts
    """
    '===== ssp madeshift ====='
    null_label = 'none'

    merge_irregular_labels = True
    irregular_prefixes = \
        {'event': ['unrelev_', 'relnscrev_', 'screv_other', 'unclear_', 'irregular', 'evoking'],
         # todo: change instance counts if evoking is excluded
         'participant': ['npart_', 'no_label', 'scrpart_other', 'suppvcomp', 'unclear', 'head_of_partitive']}
    coref1 = ['i', 'me', 'my', 'myself']
    coref2 = ['we', 'us', 'our', 'ourselves']

    irregular_event_label = '#irregularE'
    irregular_participant_label = '@irregularP'
    regular_event_label = '#regularE'
    regular_participant_label = '@regularP'

    event_label_prefix = '#'
    participant_label_prefix = '@'
    transformer_dummy_tag = 'X'

    '== Paths ==================================='
    serialization_dir = os.path.join('.', 'models')
    ssp_checkpoint_folder = \
        {'acl_ins': '/local/fangzhou/mc_script_fin/representation_checkpoints/ins_acl',
         'acl_mc': '/local/fangzhou/mc_script_fin/representation_checkpoints/ins_acl',
         # fixme
         'tacl_ins': '/local/fangzhou/mc_script_fin/representation_checkpoints/ins_acl',
         'tacl_mc': '/local/fangzhou/mc_script_fin/representation_checkpoints/mc_tacllgb2'}

    '-- data ------------------------------------'
    ins_original_dir = '/local/fangzhou/mc_script_fin/corpus/inscript_original'
    ins_train_dir = '/local/fangzhou/mc_script_fin/corpus/inscript_train'
    ins_val_dir = '/local/fangzhou/mc_script_fin/corpus/inscript_val'
    ins_test_dir = '/local/fangzhou/mc_script_fin/corpus/inscript_test'

    mcscript_all_dir = '/local/fangzhou/mc_script_fin/corpus/mc2_all'
    mcscript_train_dir = '/local/fangzhou/mc_script_fin/corpus/mc2_train'
    mcscript_val_1_dir = '/local/fangzhou/mc_script_fin/corpus/mc_val_1'
    mcscript_test_dir = '/local/fangzhou/mc_script_fin/corpus/mc_val_2'

    regularity_path = {
        'acl': {
            'ins': '/local/fangzhou/mc_script_fin/pseudo/acl/ins_acl_pseudo',
            'mc0': '/local/fangzhou/mc_script_fin/pseudo/acl/mctrain_acl_pseudo',
            'mc1': '/local/fangzhou/mc_script_fin/pseudo/acl/mc1_acl_pseudo',
            'mc2': '/local/fangzhou/mc_script_fin/pseudo/acl/mctest_acl_pseudo',
        },
        # 'tacl': {
        #     'ins': '/local/fangzhou/mc_script_fin/pseudo/acl/ins_acl_pseudo',
        #     'mc0': '/local/fangzhou/mc_script_fin/pseudo/acl/mctrain_acl_pseudo',
        #     'mc1': '/local/fangzhou/mc_script_fin/pseudo/acl/mc1_acl_pseudo',
        #     'mc2': '/local/fangzhou/mc_script_fin/pseudo/acl/mctest_acl_pseudo',
        # },
        # fixme
        'tacl': {
            'ins': '/local/fangzhou/mc_script_fin/pseudo/tacl_lg/ins_tacl_pseudo',
            'mc0': '/local/fangzhou/mc_script_fin/pseudo/tacl_lg/mcall_tacl_pseudo',
            'mc1': '/local/fangzhou/mc_script_fin/pseudo/tacl_lg/mcval_tacl_pseudo',
            'mc2': '/local/fangzhou/mc_script_fin/pseudo/tacl_lg/mctest_tacl_pseudo',
        }
    }

    '== PreProcessing ==========================='
    irregular_event_prefixes = ['unrelev_', 'relnscrev_', 'screv_other', 'unclear_', 'irregular']

    begin_of_story_event = '<story_begins>'
    begin_of_story_type = '<bost>'
    end_of_story_type = '<end_of_story>'

    '== Settings ==============================='
    pretrained_model_dim = {'xlnet-base-cased': 768, 'xlnet-large-cased': 1024}

    '==  Stats ================================='
    scenario_s = ['bath', 'bicycle', 'bus', 'cake', 'flight', 'grocery', 'haircut', 'library', 'train', 'tree']
    mcval_scenarios = ['feeding_the_fish', 'making_a_bonfire', 'cooking_pasta', 'cleaning_up_a_flat',
                       'writing_a_letter', 'going_fishing', 'taking_a_taxi', 'buying_a_house', 'playing_piano',
                       'changing_a_baby_diaper']

    mctest_scenarios = {'preparing_a_picnic', 'answering_the_phone', 'checking_in_at_an_airport', 'going_jogging', 'fueling_a_car', 'changing_a_light_bulb', 'doing_laundry', 'reading_a_story_to_a_child', 'wrapping_a_gift', 'taking_out_the_garbage'}

    all_scenario_phrases = {'bath': ['▁take', '▁a', '▁bath', '<sep>'],
                            'bicycle': ['▁fix', '▁a', '▁flat', '▁tire', '<sep>'],
                            'bus': ['▁take', '▁a', '▁bus', '<sep>'], 'cake': ['▁bake', '▁a', '▁cake', '<sep>'],
                            'flight': ['▁take', '▁a', '▁flight', '<sep>'],
                            'grocery': ['▁go', '▁grocery', '▁shopping', '<sep>'],
                            'haircut': ['▁have', '▁a', '▁hair', '▁cut', '<sep>'],
                            'library': ['▁borrow', '▁a', '▁book', '▁from', '▁library', '<sep>'],
                            'train': ['▁take', '▁a', '▁train', '<sep>'], 'tree': ['▁plant', '▁a', '▁tree', '<sep>'],
                            'washing_clothes': ['▁washing', '_', 'cloth', 'es', '<sep>'],
                            'adopting_a_pet': ['▁adopting', '_', 'a', '_', 'pet', '<sep>'],
                            'having_a_barbecue': ['▁having', '_', 'a', '_', 'barb', 'ecu', 'e', '<sep>'],
                            'playing_golf': ['▁playing', '_', 'go', 'lf', '<sep>'],
                            'changing_a_baby_diaper': ['▁changing', '_', 'a', '_', 'baby', '_', 'dia', 'per', '<sep>'],
                            'making_a_camping_trip': ['▁making', '_', 'a', '_', 'camp', 'ing', '_', 'trip', '<sep>'],
                            'drying_clothes': ['▁drying', '_', 'cloth', 'es', '<sep>'],
                            'answering_the_phone': ['▁answering', '_', 'the', '_', 'phone', '<sep>'],
                            'going_to_work': ['▁going', '_', 'to', '_', 'work', '<sep>'],
                            'taking_a_taxi': ['▁taking', '_', 'a', '_', 'tax', 'i', '<sep>'],
                            'cutting_your_own_hair': ['▁cutting', '_', 'your', '_', 'own', '_', 'hair', '<sep>'],
                            'paying_with_a_credit_card': ['▁paying', '_', 'with', '_', 'a', '_', 'credit', '_', 'card',
                                                          '<sep>'],
                            'playing_a_movie': ['▁playing', '_', 'a', '_', 'movie', '<sep>'],
                            'removing_and_replacing_a_garbage_bag': ['▁removing', '_', 'and', '_', 're', 'pla', 'cing',
                                                                     '_', 'a', '_', 'gar', 'b', 'age', '_', 'bag',
                                                                     '<sep>'],
                            'unclogging_the_toilet': ['▁', 'unc', 'logging', '_', 'the', '_', 'to', 'ile', 't',
                                                      '<sep>'],
                            'mowing_the_lawn': ['▁mo', 'wing', '_', 'the', '_', 'law', 'n', '<sep>'],
                            'vacuuming_the_carpet': ['▁vacuum', 'ing', '_', 'the', '_', 'car', 'pet', '<sep>'],
                            'making_a_bed': ['▁making', '_', 'a', '_', 'bed', '<sep>'],
                            'taking_a_bath': ['▁taking', '_', 'a', '_', 'bath', '<sep>'],
                            'heating_food_on_kitchen_gas': ['▁heating', '_', 'food', '_', 'on', '_', 'kit', 'chen', '_',
                                                            'gas', '<sep>'],
                            'going_to_a_pub': ['▁going', '_', 'to', '_', 'a', '_', 'pub', '<sep>'],
                            'going_on_vacation_or_going_on_a_holiday_trip': ['▁going', '_', 'on', '_', 'vac', 'ation',
                                                                             '_', 'or', '_', 'going', '_', 'on', '_',
                                                                             'a', '_', 'hol', 'i', 'day', '_', 'trip',
                                                                             '<sep>'],
                            'training_a_dog': ['▁training', '_', 'a', '_', 'dog', '<sep>'],
                            'checking_if_a_store_is_open': ['▁checking', '_', 'if', '_', 'a', '_', 'store', '_', 'is',
                                                            '_', 'open', '<sep>'],
                            'shopping_for_clothes': ['▁shopping', '_', 'for', '_', 'cloth', 'es', '<sep>'],
                            'walking_a_dog': ['▁walking', '_', 'a', '_', 'dog', '<sep>'],
                            'cooking_rice': ['▁cooking', '_', 'rice', '<sep>'],
                            'planning_a_holiday_trip': ['▁planning', '_', 'a', '_', 'hol', 'i', 'day', '_', 'trip',
                                                        '<sep>'],
                            'making_a_bonfire': ['▁making', '_', 'a', '_', 'bon', 'fire', '<sep>'],
                            'loading_the_dishwasher': ['▁loading', '_', 'the', '_', 'd', 'ish', 'wash', 'er', '<sep>'],
                            'going_grocery_shopping': ['▁going', '_', 'gro', 'cer', 'y', '_', 's', 'hopping', '<sep>'],
                            'changing_batteries_in_an_alarm_clock': ['▁changing', '_', 'bat', 't', 'eries', '_', 'in',
                                                                     '_', 'an', '_', 'al', 'arm', '_', 'clock',
                                                                     '<sep>'],
                            'taking_copies': ['▁taking', '_', 'co', 'pies', '<sep>'],
                            'making_fresh_orange_juice': ['▁making', '_', 'fresh', '_', 'o', 'range', '_', 'ju', 'ice',
                                                          '<sep>'],
                            'cooking_pasta': ['▁cooking', '_', 'pa', 'sta', '<sep>'],
                            'buying_a_tree': ['▁buying', '_', 'a', '_', 'tree', '<sep>'],
                            'moving_into_a_new_flat': ['▁moving', '_', 'in', 'to', '_', 'a', '_', 'new', '_', 'flat',
                                                       '<sep>'],
                            'watering_indoor_plants': ['▁water', 'ing', '_', 'in', 'door', '_', 'plant', 's', '<sep>'],
                            'making_a_shopping_list': ['▁making', '_', 'a', '_', 's', 'hopping', '_', 'list', '<sep>'],
                            'making_toasted_bread': ['▁making', '_', 'to', 'as', 'ted', '_', 'bread', '<sep>'],
                            'eating_in_a_fast_food_restaurant': ['▁eating', '_', 'in', '_', 'a', '_', 'fast', '_',
                                                                 'food', '_', 'rest', 'au', 'rant', '<sep>'],
                            'going_skiing': ['▁going', '_', 'ski', 'ing', '<sep>'],
                            'changing_a_light_bulb': ['▁changing', '_', 'a', '_', 'light', '_', 'bul', 'b', '<sep>'],
                            'listening_to_music': ['▁listening', '_', 'to', '_', 'music', '<sep>'],
                            'going_for_a_walk': ['▁going', '_', 'for', '_', 'a', '_', 'walk', '<sep>'],
                            'making_a_flight_reservation': ['▁making', '_', 'a', '_', 'flight', '_', 're', 'serv',
                                                            'ation', '<sep>'],
                            'doing_laundry': ['▁doing', '_', 'la', 'und', 'ry', '<sep>'],
                            'checking_in_at_an_airport': ['▁checking', '_', 'in', '_', 'at', '_', 'an', '_', 'air',
                                                          'port', '<sep>'],
                            'taking_care_of_children': ['▁taking', '_', 'care', '_', 'of', '_', 'children', '<sep>'],
                            'moving_furniture': ['▁moving', '_', 'fur', 'ni', 'ture', '<sep>'],
                            'emptying_the_kitchen_sink': ['▁empty', 'ing', '_', 'the', '_', 'kit', 'chen', '_', 's',
                                                          'ink', '<sep>'],
                            'preparing_a_wedding': ['▁preparing', '_', 'a', '_', 'we', 'dding', '<sep>'],
                            'answering_the_doorbell': ['▁answering', '_', 'the', '_', 'door', 'bell', '<sep>'],
                            'setting_up_presentation_equipment': ['▁setting', '_', 'up', '_', 'present', 'ation', '_',
                                                                  'equi', 'p', 'ment', '<sep>'],
                            'chopping_vegetables': ['▁chop', 'ping', '_', 've', 'get', 'able', 's', '<sep>'],
                            'washing_a_cut': ['▁washing', '_', 'a', '_', 'cut', '<sep>'],
                            'going_on_a_train': ['▁going', '_', 'on', '_', 'a', '_', 'train', '<sep>'],
                            'repairing_a_bicycle': ['▁repairing', '_', 'a', '_', 'bi', 'cycle', '<sep>'],
                            'visiting_relatives': ['▁visiting', '_', 'rel', 'ative', 's', '<sep>'],
                            'buying_from_a_vending_machine': ['▁buying', '_', 'from', '_', 'a', '_', 'v', 'ending', '_',
                                                              'machine', '<sep>'],
                            'throwing_a_party': ['▁throwing', '_', 'a', '_', 'party', '<sep>'],
                            'painting_a_wall': ['▁painting', '_', 'a', '_', 'wall', '<sep>'],
                            'taking_children_to_school': ['▁taking', '_', 'children', '_', 'to', '_', 'school',
                                                          '<sep>'],
                            'making_coffee': ['▁making', '_', 'co', 'ffe', 'e', '<sep>'],
                            'putting_up_a_painting': ['▁putting', '_', 'up', '_', 'a', '_', 'pa', 'in', 'ting',
                                                      '<sep>'],
                            'folding_clothes': ['▁folding', '_', 'cloth', 'es', '<sep>'],
                            'going_jogging': ['▁going', '_', 'jo', 'gging', '<sep>'],
                            'flying_in_a_plane': ['▁flying', '_', 'in', '_', 'a', '_', 'plane', '<sep>'],
                            'looking_for_a_flat': ['▁looking', '_', 'for', '_', 'a', '_', 'flat', '<sep>'],
                            'taking_a_photograph': ['▁taking', '_', 'a', '_', 'photo', 'graph', '<sep>'],
                            'putting_a_poster_on_the_wall': ['▁putting', '_', 'a', '_', 'post', 'er', '_', 'on', '_',
                                                             'the', '_', 'wall', '<sep>'],
                            'ordering_a_pizza': ['▁ordering', '_', 'a', '_', 'pi', 'zza', '<sep>'],
                            'paying_taxes': ['▁paying', '_', 'tax', 'es', '<sep>'],
                            'taking_out_the_garbage': ['▁taking', '_', 'out', '_', 'the', '_', 'gar', 'b', 'age',
                                                       '<sep>'],
                            'cleaning_up_a_flat': ['▁cleaning', '_', 'up', '_', 'a', '_', 'flat', '<sep>'],
                            'taking_a_child_to_bed': ['▁taking', '_', 'a', '_', 'child', '_', 'to', '_', 'bed',
                                                      '<sep>'],
                            'serving_a_meal': ['▁serving', '_', 'a', '_', 'me', 'al', '<sep>'],
                            'cleaning_up_toys': ['▁cleaning', '_', 'up', '_', 'to', 'y', 's', '<sep>'],
                            'applying_band_aid': ['▁applying', '_', 'band', '_', 'aid', '<sep>'],
                            'reading_a_story_to_a_child': ['▁reading', '_', 'a', '_', 'story', '_', 'to', '_', 'a', '_',
                                                           'child', '<sep>'],
                            'cleaning_the_floor': ['▁cleaning', '_', 'the', '_', 'floor', '<sep>'],
                            'designing_t-shirts': ['▁designing', '_', 't', '-', 'shirts', '<sep>'],
                            'sending_food_back_(in_a_restaurant)': ['▁sending', '_', 'food', '_', 'back', '_', '(',
                                                                    'in', '_', 'a', '_', 'rest', 'au', 'rant', ')',
                                                                    '<sep>'],
                            'getting_a_haircut': ['▁getting', '_', 'a', '_', 'hair', 'cut', '<sep>'],
                            'driving_a_car': ['▁driving', '_', 'a', '_', 'car', '<sep>'],
                            'visiting_a_museum': ['▁visiting', '_', 'a', '_', 'museum', '<sep>'],
                            'attending_a_court_hearing': ['▁attending', '_', 'a', '_', 'court', '_', 'hear', 'ing',
                                                          '<sep>'],
                            'going_to_a_party': ['▁going', '_', 'to', '_', 'a', '_', 'party', '<sep>'],
                            'reviewing_movies': ['▁reviewing', '_', 'movie', 's', '<sep>'],
                            'ironing_laundry': ['▁iron', 'ing', '_', 'la', 'und', 'ry', '<sep>'],
                            'heating_food_in_a_microwave': ['▁heating', '_', 'food', '_', 'in', '_', 'a', '_', 'micro',
                                                            'wave', '<sep>'],
                            'preparing_dinner': ['▁preparing', '_', 'din', 'ner', '<sep>'],
                            'taking_the_underground': ['▁taking', '_', 'the', '_', 'under', 'ground', '<sep>'],
                            'watching_a_tennis_match': ['▁watching', '_', 'a', '_', 'ten', 'nis', '_', 'match',
                                                        '<sep>'],
                            'setting_an_alarm': ['▁setting', '_', 'an', '_', 'al', 'arm', '<sep>'],
                            'going_bowling': ['▁going', '_', 'bow', 'ling', '<sep>'],
                            'playing_tennis': ['▁playing', '_', 'ten', 'nis', '<sep>'],
                            'renovating_a_room': ['▁', 're', 'nova', 'ting', '_', 'a', '_', 'room', '<sep>'],
                            'visiting_sights': ['▁visiting', '_', 's', 'ight', 's', '<sep>'],
                            'renting_a_movie': ['▁renting', '_', 'a', '_', 'movie', '<sep>'],
                            'going_to_a_concert': ['▁going', '_', 'to', '_', 'a', '_', 'con', 'cer', 't', '<sep>'],
                            'washing_dishes': ['▁washing', '_', 'd', 'ish', 'es', '<sep>'],
                            'cleaning_the_bathroom': ['▁cleaning', '_', 'the', '_', 'bath', 'room', '<sep>'],
                            'wrapping_a_gift': ['▁wrapping', '_', 'a', '_', 'gi', 'ft', '<sep>'],
                            'visiting_the_beach': ['▁visiting', '_', 'the', '_', 'be', 'ach', '<sep>'],
                            'attending_a_church_service': ['▁attending', '_', 'a', '_', 'church', '_', 'service',
                                                           '<sep>'],
                            'growing_vegetables': ['▁growing', '_', 've', 'get', 'able', 's', '<sep>'],
                            'attending_a_job_interview': ['▁attending', '_', 'a', '_', 'job', '_', 'inter', 'view',
                                                          '<sep>'],
                            'sewing_a_button': ['▁sewing', '_', 'a', '_', 'button', '<sep>'],
                            'settling_bank_transactions': ['▁settling', '_', 'bank', '_', 'trans', 'action', 's',
                                                           '<sep>'],
                            'packing_a_suitcase': ['▁packing', '_', 'a', '_', 'suit', 'case', '<sep>'],
                            'calling_911': ['▁calling', '_', '9', '11', '<sep>'],
                            'going_to_the_sauna': ['▁going', '_', 'to', '_', 'the', '_', 'sa', 'una', '<sep>'],
                            'playing_music_in_church': ['▁playing', '_', 'music', '_', 'in', '_', 'church', '<sep>'],
                            'making_a_hot_dog': ['▁making', '_', 'a', '_', 'hot', '_', 'dog', '<sep>'],
                            'making_a_dinner_reservation': ['▁making', '_', 'a', '_', 'din', 'ner', '_', 're', 'serv',
                                                            'ation', '<sep>'],
                            'taking_a_swimming_class': ['▁taking', '_', 'a', '_', 's', 'wi', 'mm', 'ing', '_', 'class',
                                                        '<sep>'],
                            'feeding_a_cat': ['▁feeding', '_', 'a', '_', 'cat', '<sep>'],
                            'looking_for_a_job': ['▁looking', '_', 'for', '_', 'a', '_', 'job', '<sep>'],
                            'borrowing_a_book_from_the_library': ['▁borrowing', '_', 'a', '_', 'book', '_', 'from', '_',
                                                                  'the', '_', 'li', 'br', 'ary', '<sep>'],
                            'buying_a_DVD_player': ['▁buying', '_', 'a', '_', 'DVD', '_', 'player', '<sep>'],
                            'going_to_a_shopping_centre': ['▁going', '_', 'to', '_', 'a', '_', 's', 'hopping', '_',
                                                           'centre', '<sep>'],
                            'working_in_the_garden': ['▁working', '_', 'in', '_', 'the', '_', 'garden', '<sep>'],
                            'brushing_teeth': ['▁brushing', '_', 'tee', 'th', '<sep>'],
                            'making_soup': ['▁making', '_', 'so', 'up', '<sep>'],
                            'going_shopping': ['▁going', '_', 's', 'hopping', '<sep>'],
                            'organizing_a_board_game_evening': ['▁organizing', '_', 'a', '_', 'board', '_', 'game', '_',
                                                                'even', 'ing', '<sep>'],
                            'sending_a_fax': ['▁sending', '_', 'a', '_', 'fax', '<sep>'],
                            'cooking_meat': ['▁cooking', '_', 'me', 'at', '<sep>'],
                            'planting_flowers': ['▁planting', '_', 'flower', 's', '<sep>'],
                            'making_a_sandwich': ['▁making', '_', 'a', '_', 's', 'and', 'wich', '<sep>'],
                            'going_to_the_swimming_pool': ['▁going', '_', 'to', '_', 'the', '_', 's', 'wi', 'mm', 'ing',
                                                           '_', 'pool', '<sep>'],
                            'receiving_a_letter': ['▁receiving', '_', 'a', '_', 'letter', '<sep>'],
                            'riding_on_a_bus': ['▁riding', '_', 'on', '_', 'a', '_', 'bus', '<sep>'],
                            'giving_a_medicine_to_someone': ['▁giving', '_', 'a', '_', 'med', 'ic', 'ine', '_', 'to',
                                                             '_', 'some', 'one', '<sep>'],
                            'going_to_the_gym': ['▁going', '_', 'to', '_', 'the', '_', 'gy', 'm', '<sep>'],
                            'paying_for_gas': ['▁paying', '_', 'for', '_', 'gas', '<sep>'],
                            'getting_ready_for_bed': ['▁getting', '_', 'ready', '_', 'for', '_', 'bed', '<sep>'],
                            'planting_a_tree': ['▁planting', '_', 'a', '_', 'tree', '<sep>'],
                            'making_tea': ['▁making', '_', 'tea', '<sep>'],
                            'playing_football': ['▁playing', '_', 'foot', 'ball', '<sep>'],
                            'baking_a_cake': ['▁baking', '_', 'a', '_', 'cake', '<sep>'],
                            'going_dancing': ['▁going', '_', 'd', 'ancing', '<sep>'],
                            'serving_a_drink': ['▁serving', '_', 'a', '_', 'd', 'rink', '<sep>'],
                            'shopping_online': ['▁shopping', '_', 'online', '<sep>'],
                            'buying_a_birthday_present': ['▁buying', '_', 'a', '_', 'birth', 'day', '_', 'present',
                                                          '<sep>'],
                            'putting_away_groceries': ['▁putting', '_', 'away', '_', 'gro', 'ce', 'ries', '<sep>'],
                            'cleaning_the_shower': ['▁cleaning', '_', 'the', '_', 'show', 'er', '<sep>'],
                            'ordering_something_on_the_phone': ['▁ordering', '_', 'something', '_', 'on', '_', 'the',
                                                                '_', 'phone', '<sep>'],
                            'telling_a_story': ['▁telling', '_', 'a', '_', 'story', '<sep>'],
                            'making_omelette': ['▁making', '_', 'ome', 'lette', '<sep>'],
                            'deciding_on_a_movie': ['▁deciding', '_', 'on', '_', 'a', '_', 'movie', '<sep>'],
                            'going_to_the_theater': ['▁going', '_', 'to', '_', 'the', '_', 'th', 'eater', '<sep>'],
                            'cooking_fish': ['▁cooking', '_', 'fish', '<sep>'],
                            'making_scrambled_eggs': ['▁making', '_', 's', 'cra', 'mble', 'd', '_', 'e', 'gg', 's',
                                                      '<sep>'],
                            'getting_the_newspaper': ['▁getting', '_', 'the', '_', 'news', 'paper', '<sep>'],
                            'feeding_an_infant': ['▁feeding', '_', 'an', '_', 'in', 'fan', 't', '<sep>'],
                            'cleaning_a_kitchen': ['▁cleaning', '_', 'a', '_', 'kit', 'chen', '<sep>'],
                            'playing_video_games': ['▁playing', '_', 'video', '_', 'games', '<sep>'],
                            'laying_flooring_in_a_room': ['▁laying', '_', 'floor', 'ing', '_', 'in', '_', 'a', '_',
                                                          'room', '<sep>'],
                            'going_on_a_date': ['▁going', '_', 'on', '_', 'a', '_', 'date', '<sep>'],
                            'preparing_a_picnic': ['▁preparing', '_', 'a', '_', 'pic', 'nic', '<sep>'],
                            'sewing_clothes': ['▁sewing', '_', 'cloth', 'es', '<sep>'],
                            'attending_a_wedding_ceremony': ['▁attending', '_', 'a', '_', 'we', 'dding', '_', 'ce',
                                                             're', 'mony', '<sep>'],
                            'playing_a_board_game': ['▁playing', '_', 'a', '_', 'board', '_', 'game', '<sep>'],
                            'going_to_a_funeral': ['▁going', '_', 'to', '_', 'a', '_', 'fun', 'er', 'al', '<sep>'],
                            'changing_bed_sheets': ['▁changing', '_', 'bed', '_', 'sheet', 's', '<sep>'],
                            'washing_one’s_hair': ['▁washing', '_', 'one', '’', 's', '_', 'hair', '<sep>'],
                            'making_a_mixed_salad': ['▁making', '_', 'a', '_', 'mix', 'ed', '_', 'sal', 'ad', '<sep>'],
                            'writing_an_exam': ['▁writing', '_', 'an', '_', 'ex', 'am', '<sep>'],
                            'feeding_the_fish': ['▁feeding', '_', 'the', '_', 'fish', '<sep>'],
                            'locking_up_the_house': ['▁locking', '_', 'up', '_', 'the', '_', 'house', '<sep>'],
                            'playing_a_song': ['▁playing', '_', 'a', '_', 's', 'ong', '<sep>'],
                            'unloading_the_dishwasher': ['▁unload', 'ing', '_', 'the', '_', 'd', 'ish', 'wash', 'er',
                                                         '<sep>'],
                            'visiting_a_doctor': ['▁visiting', '_', 'a', '_', 'do', 'ctor', '<sep>'],
                            'taking_a_driving_lesson': ['▁taking', '_', 'a', '_', 'driving', '_', 'less', 'on',
                                                        '<sep>'],
                            'playing_piano': ['▁playing', '_', 'pian', 'o', '<sep>'],
                            'canceling_a_party': ['▁cancel', 'ing', '_', 'a', '_', 'party', '<sep>'],
                            'boiling_milk': ['▁boiling', '_', 'milk', '<sep>'],
                            'cleaning_the_table': ['▁cleaning', '_', 'the', '_', 'table', '<sep>'],
                            'taking_a_shower': ['▁taking', '_', 'a', '_', 'show', 'er', '<sep>'],
                            'learning_a_board_game': ['▁learning', '_', 'a', '_', 'board', '_', 'game', '<sep>'],
                            'mailing_a_letter': ['▁mailing', '_', 'a', '_', 'letter', '<sep>'],
                            'going_on_a_bike_tour': ['▁going', '_', 'on', '_', 'a', '_', 'bi', 'ke', '_', 'tour',
                                                     '<sep>'],
                            'attending_a_football_match': ['▁attending', '_', 'a', '_', 'foot', 'ball', '_', 'match',
                                                           '<sep>'],
                            'going_to_the_playground': ['▁going', '_', 'to', '_', 'the', '_', 'play', 'ground',
                                                        '<sep>'],
                            'papering_a_room': ['▁paper', 'ing', '_', 'a', '_', 'room', '<sep>'],
                            'eating_in_a_restaurant': ['▁eating', '_', 'in', '_', 'a', '_', 'rest', 'au', 'rant',
                                                       '<sep>'],
                            'sending_party_invitations': ['▁sending', '_', 'party', '_', 'in', 'vit', 'ations',
                                                          '<sep>'],
                            'doing_online_banking': ['▁doing', '_', 'online', '_', 'bank', 'ing', '<sep>'],
                            'fueling_a_car': ['▁fuel', 'ing', '_', 'a', '_', 'car', '<sep>'],
                            'taking_medicine': ['▁taking', '_', 'med', 'ic', 'ine', '<sep>'],
                            'changing_batteries_in_a_camera': ['▁changing', '_', 'bat', 't', 'eries', '_', 'in', '_',
                                                               'a', '_', 'camera', '<sep>'],
                            'going_to_the_dentist': ['▁going', '_', 'to', '_', 'the', '_', 'dent', 'ist', '<sep>'],
                            'going_fishing': ['▁going', '_', 'fish', 'ing', '<sep>'],
                            'paying_bills': ['▁paying', '_', 'bill', 's', '<sep>'],
                            'making_breakfast': ['▁making', '_', 'break', 'fast', '<sep>'],
                            'buying_a_house': ['▁buying', '_', 'a', '_', 'house', '<sep>'],
                            'writing_a_letter': ['▁writing', '_', 'a', '_', 'letter', '<sep>'],
                            'setting_the_dining_table': ['▁setting', '_', 'the', '_', 'din', 'ing', '_', 'table',
                                                         '<sep>']}

    ssp_model_indices = \
        {
            'bath': 12,
            'bicycle': 1,
            'bus': 4,
            'cake': 0,
            'flight': 4,
            'grocery': 4,
            'haircut': 3,
            'library': 11,
            'train': 3,
            'tree': 0
        }
    #
    # scenario_instance_counts = \
    #     {
    #         'bath': 8209,
    #         'bicycle': 7770,
    #         'bus': 8589,
    #         'cake': 10061,
    #         'flight': 9600,
    #         'grocery': 10098,
    #         'haircut': 9694,
    #         'library': 8575,
    #         'train': 8283,
    #         'tree': 8111
    #     }

    n_event_cluster = \
        {
            'bath': 20,
            'bicycle': 16,
            'bus': 17,
            'cake': 19,
            'flight': 29,
            'grocery': 19,
            'haircut': 26,
            'library': 17,
            'train': 15,
            'tree': 13
        }

    n_participant_cluster = \
        {
            'bath': 19,
            'bicycle': 17,
            'bus': 18,
            'cake': 18,
            'flight': 27,
            'grocery': 19,
            'haircut': 25,
            'library': 19,
            'train': 21,
            'tree': 16
        }

    n_classes = 392

    effective_scenario_s = ['bath']

    #
    begin_of_sequence = '<bose>'
    end_of_sequence = '<eose>'
    dummy_event_annotation_prefix = '<no_event_annotated>'
    include_protagonists = False

#  sum([len(inst.fields['participant_labels'].tokens) for inst in gt_instances])
    # ''' this is without prota / evoking '''
    # ins_gt_merged_index_count = {'bath': 3507, 'bicycle': 3475, 'bus': 3731, 'cake': 5765, 'flight': 4654, 'grocery': 5009, 'haircut': 4915, 'library': 3786, 'train': 3887, 'tree': 3716}
    # mc1_gt_merged_index_count = {'feeding_the_fish': 618, 'making_a_bonfire': 863, 'cooking_pasta': 968, 'cleaning_up_a_flat': 660,
    #  'writing_a_letter': 742, 'going_fishing': 781, 'taking_a_taxi': 781, 'buying_a_house': 640, 'playing_piano': 472,
    #  'changing_a_baby_diaper': 833}
    # mc2_gt_merged_index_count = {'preparing_a_picnic': 850, 'changing_a_light_bulb': 763, 'taking_out_the_garbage': 845, 'going_jogging': 547, 'reading_a_story_to_a_child': 710, 'fueling_a_car': 978, 'checking_in_at_an_airport': 650, 'wrapping_a_gift': 915, 'answering_the_phone': 665, 'doing_laundry': 758}
    # mc_all_merged_index_count = {'preparing_a_picnic': 850, 'changing_a_light_bulb': 763, 'taking_out_the_garbage': 845, 'going_jogging': 547, 'reading_a_story_to_a_child': 710, 'fueling_a_car': 978, 'checking_in_at_an_airport': 650, 'wrapping_a_gift': 915, 'answering_the_phone': 665, 'doing_laundry': 758, 'feeding_the_fish': 618, 'making_a_bonfire': 863, 'cooking_pasta': 968, 'cleaning_up_a_flat': 660,
    #  'writing_a_letter': 742, 'going_fishing': 781, 'taking_a_taxi': 781, 'buying_a_house': 640, 'playing_piano': 472,
    #  'changing_a_baby_diaper': 833}
    # # fixme
    # ''' this is without prota, with evoking '''
    # # ins_gt_merged_index_count = {'bath': 3809, 'bicycle': 3727, 'bus': 4043, 'cake': 6184, 'flight': 4936, 'grocery': 5217, 'haircut': 5234, 'library': 4110, 'train': 4205, 'tree': 4028}
    # # mc1_gt_merged_index_count = {'feeding_the_fish': 648, 'making_a_bonfire': 883, 'cooking_pasta': 993,
    # #                              'cleaning_up_a_flat': 678, 'writing_a_letter': 760, 'going_fishing': 814,
    # #                              'taking_a_taxi': 791, 'buying_a_house': 660, 'playing_piano': 504,
    # #                              'changing_a_baby_diaper': 856}
    # # mc2_gt_merged_index_count = {'preparing_a_picnic': 860, 'answering_the_phone': 677, 'checking_in_at_an_airport': 661, 'going_jogging': 563, 'fueling_a_car': 987, 'changing_a_light_bulb': 774, 'doing_laundry': 772, 'reading_a_story_to_a_child': 726, 'wrapping_a_gift': 937, 'taking_out_the_garbage': 872}
    # # mc_all_merged_index_count = {'preparing_a_picnic': 860, 'answering_the_phone': 677,
    # #                              'checking_in_at_an_airport': 661, 'going_jogging': 563, 'fueling_a_car': 987,
    # #                              'changing_a_light_bulb': 774, 'doing_laundry': 772, 'reading_a_story_to_a_child': 726,
    # #                              'wrapping_a_gift': 937, 'taking_out_the_garbage': 872, 'feeding_the_fish': 648,
    # #                              'making_a_bonfire': 883, 'cooking_pasta': 993, 'cleaning_up_a_flat': 678,
    # #                              'writing_a_letter': 760, 'going_fishing': 814, 'taking_a_taxi': 791,
    # #                              'buying_a_house': 660, 'playing_piano': 504, 'changing_a_baby_diaper': 856}
    #
    # # ins_gt_merged_index_count = {'bath': 3852, 'bicycle': 3654, 'bus': 4060, 'cake': 6053, 'flight': 4947, 'grocery': 5272, 'haircut': 5158, 'library': 4002, 'train': 4112, 'tree': 3930}
    # # ''' this is without prota '''
    # # # ins_gt_merged_index_count = {'bath': 3728, 'bicycle': 3554, 'bus': 3973, 'cake': 5901, 'flight': 4878, 'grocery': 5196, 'haircut': 5042, 'library': 3898, 'train': 4022, 'tree': 3779}
    # # #  {'bath': 3822, 'bicycle': 3593, 'bus': 4009, 'cake': 5981, 'flight': 4916, 'grocery': 5315, 'haircut': 5102, 'library': 3972, 'train': 4078, 'tree': 3853}
    # # bt_gt_merged_index_count = {'bath': 7478, 'bicycle': 6874, 'bus': 7469, 'cake': 10938, 'flight': 8987, 'grocery': 9768, 'haircut': 9395, 'library': 7511, 'train': 7522, 'tree': 6809}
    # # ''' this is with prota '''
    # # # ins_gt_merged_index_count = {'bath': 5705, 'bicycle': 5016, 'bus': 5604, 'cake': 7402, 'flight': 6627, 'grocery': 7368, 'haircut': 6981, 'library': 5503, 'train': 5388, 'tree': 4943}
    # # # with prota: {'bath': 9027, 'bicycle': 8044, 'bus': 8726, 'cake': 12078, 'flight': 10344, 'grocery': 11441, 'haircut': 10912, 'library': 8736, 'train': 8575, 'tree': 7708}

