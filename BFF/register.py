from best_friend_forever import User
from best_friend_forever import load_users
from generate_users import save_user_data


# Attention: After run this program for testing, run generate_users.py to reset user data list.
if __name__ == '__main__':
    user_list = load_users()
    new_user = User(len(user_list))
    user_list.append(new_user)
    save_user_data(new_user, len(user_list)+1, 0)
    print("Your ID is "+str(len(user_list)-1)+", welcome!")
