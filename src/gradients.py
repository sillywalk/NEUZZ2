
# compute gradient for given input


def gen_adv2(f, fl, model, layer_list, idxx, splice):
    adv_list = []
    loss = layer_list[-2][1].output[:, f]
    grads = K.gradients(loss, model.input)[0]
    iterate = K.function([model.input], [loss, grads])
    ll = 2
    while(fl[0] == fl[1]):
        fl[1] = random.choice(seed_list)

    for index in range(ll):
        x = vectorize_file(fl[index])
        loss_value, grads_value = iterate([x])
        idx = np.flip(np.argsort(np.absolute(grads_value), axis=1)[
                      :, -MAX_FILE_SIZE:].reshape((MAX_FILE_SIZE,)), 0)
        val = np.sign(grads_value[0][idx])
        adv_list.append((idx, val, fl[index]))

    if(splice == 1):
        # do not generate spliced seed for the first round
        if(round_cnt != 0):
            if(round_cnt % 2 == 0):
                splice_seed(fl[0], fl[1], idxx)
                x = vectorize_file('./splice_seeds/tmp_' + str(idxx))
                loss_value, grads_value = iterate([x])
                idx = np.flip(np.argsort(np.absolute(grads_value), axis=1)[
                              :, -MAX_FILE_SIZE:].reshape((MAX_FILE_SIZE,)), 0)
                val = np.sign(grads_value[0][idx])
                adv_list.append((idx, val, './splice_seeds/tmp_' + str(idxx)))
            else:
                splice_seed(fl[0], fl[1], idxx + 500)
                x = vectorize_file('./splice_seeds/tmp_' + str(idxx + 500))
                loss_value, grads_value = iterate([x])
                idx = np.flip(np.argsort(np.absolute(grads_value), axis=1)[
                              :, -MAX_FILE_SIZE:].reshape((MAX_FILE_SIZE,)), 0)
                val = np.sign(grads_value[0][idx])
                adv_list.append(
                    (idx, val, './splice_seeds/tmp_' + str(idxx + 500)))

    return adv_list

# compute gradient for given input without sign


def gen_adv3(f, fl, model, layer_list, idxx, splice):
    adv_list = []
    loss = layer_list[-2][1].output[:, f]
    grads = K.gradients(loss, model.input)[0]
    iterate = K.function([model.input], [loss, grads])
    ll = 2
    while(fl[0] == fl[1]):
        fl[1] = random.choice(seed_list)

    for index in range(ll):
        x = vectorize_file(fl[index])
        loss_value, grads_value = iterate([x])
        idx = np.flip(np.argsort(np.absolute(grads_value), axis=1)[
                      :, -MAX_FILE_SIZE:].reshape((MAX_FILE_SIZE,)), 0)
        #val = np.sign(grads_value[0][idx])
        val = np.random.choice([1, -1], MAX_FILE_SIZE, replace=True)
        adv_list.append((idx, val, fl[index]))

    if(splice == 1):
        # do not generate spliced seed for the first round
        if(round_cnt != 0):
            splice_seed(fl[0], fl[1], idxx)
            x = vectorize_file('./splice_seeds/tmp_' + str(idxx))
            loss_value, grads_value = iterate([x])
            idx = np.flip(np.argsort(np.absolute(grads_value), axis=1)[
                          :, -MAX_FILE_SIZE:].reshape((MAX_FILE_SIZE,)), 0)
            #val = np.sign(grads_value[0][idx])
            val = np.random.choice([1, -1], MAX_FILE_SIZE, replace=True)
            adv_list.append((idx, val, './splice_seeds/tmp_' + str(idxx)))

    return adv_list

# grenerate gradient information to guide furture muatation


def gen_mutate2(model, edge_num, sign):
    tmp_list = []
    # select seeds
    print("#######debug" + str(round_cnt))
    if(round_cnt == 0):
        new_seed_list = seed_list
    else:
        new_seed_list = new_seeds

    if len(new_seed_list) < edge_num:
        rand_seed1 = [new_seed_list[i] for i in np.random.choice(
            len(new_seed_list), edge_num, replace=True)]
    else:
        rand_seed1 = [new_seed_list[i] for i in np.random.choice(
            len(new_seed_list), edge_num, replace=False)]
    if len(new_seed_list) < edge_num:
        rand_seed2 = [seed_list[i] for i in np.random.choice(
            len(seed_list), edge_num, replace=True)]
    else:
        rand_seed2 = [seed_list[i] for i in np.random.choice(
            len(seed_list), edge_num, replace=False)]

    # function pointer for gradient computation
    fn = gen_adv2
    if (sign):
        fn = gen_adv2
    else:
        fn = gen_adv3

    # select output neurons to compute gradient
    interested_indice = np.random.choice(MAX_BITMAP_SIZE, edge_num)
    layer_list = [(layer.name, layer) for layer in model.layers]

    with open('gradient_info_p', 'w') as f:
        for idxx in range(len(interested_indice[:])):
            # kears's would stall after multiple gradient compuation. Release memory and reload model to fix it.
            if (idxx % 100 == 0):
                del model
                K.clear_session()
                model = build_model()
                model.load_weights('hard_label.h5')
                layer_list = [(layer.name, layer) for layer in model.layers]

            print("number of feature " + str(idxx))
            index = int(interested_indice[idxx])
            fl = [rand_seed1[idxx], rand_seed2[idxx]]
            adv_list = fn(index, fl, model, layer_list, idxx, 1)
            tmp_list.append(adv_list)
            for ele in adv_list:
                ele0 = [str(el) for el in ele[0]]
                ele1 = [str(int(el)) for el in ele[1]]
                ele2 = ele[2]
                f.write(",".join(ele0) + '|' +
                        ",".join(ele1) + '|' + ele2 + "\n")


def gen_grad(data):
    global round_cnt
    t0 = time.time()
    process_data()
    model = build_model()
    train(model)
    #model.load_weights('hard_label.h5')
    if(data[:5] == "train"):
        gen_mutate2(model, 500, True)
    else:
        gen_mutate2(model, 500, False)
    round_cnt = round_cnt + 1
    print(time.time() - t0)
