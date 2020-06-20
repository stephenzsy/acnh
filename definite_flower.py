from functools import reduce

class Flower:
                
    def get_index(self, l):
        return reduce(lambda a, b: a * 3 + b, l, 0)
    
    def __init__(self, name, n, m):
        self.name = name
        self.n = n
        self.m = m
        self.cm = {}
        
        for color, tl in m.items():
            for t in tl:
                assert len(t) == self.n
                for e in t:
                    assert e < 3
                index = self.get_index(t)
                assert index not in self.cm
                self.cm[self.get_index(t)] = color
                
        tc = 3 ** self.n
        ec = sum(map(lambda a: len(a), self.m.values()))
        assert tc == ec
            
    def get_tuple(self, l):
        r = [0] * self.n
        for i in range(self.n - 1, -1, -1):
            r[i] = l % 3
            l //= 3
        return r
    
    def mate(self, index1, index2):        
        def mate_inner(t1, t2, c, p, i):
            if i >= self.n:
                return [(self.get_index(c), p)]
            n1, n2 = t1[i], t2[i]
            r = []
            if n1 == 0:
                if n2 == 0:
                    c[i] = 0
                    r += mate_inner(t1, t2, c, p, i + 1)
                elif n2 == 1:
                    c[i] = 0
                    r += mate_inner(t1, t2, c, p / 2, i + 1)
                    c[i] = 1
                    r += mate_inner(t1, t2, c, p / 2, i + 1)
                elif n2 == 2:
                    c[i] = 1
                    r += mate_inner(t1, t2, c, p, i + 1)
            elif n1 == 1:
                if n2 == 1:
                    c[i] = 0
                    r += mate_inner(t1, t2, c, p / 4, i + 1)
                    c[i] = 1
                    r += mate_inner(t1, t2, c, p / 2, i + 1)
                    c[i] = 2
                    r += mate_inner(t1, t2, c, p / 4, i + 1)
                elif n2 == 2:
                    c[i] = 1
                    r += mate_inner(t1, t2, c, p / 2, i + 1)
                    c[i] = 2
                    r += mate_inner(t1, t2, c, p / 2, i + 1)
            elif n1 == 2:
                if n2 == 2:
                    c[i] = 2
                    r += mate_inner(t1, t2, c, p, i + 1)

            return r
        return mate_inner(self.get_tuple(index1), self.get_tuple(index2), [None] * self.n, 1.0, 0)
    
    def mate_definite(self, index1, index2):
        [i1, i2] = sorted([index1, index2])
        r = self.mate(i1, i2)
        m = {}
        results = []
        for t in r:
            i, p = t
            c = self.cm[i]
            if c not in m:
                m[c] = []
            m[c].append((i, p))
        for c, t in m.items():
            if len(t) == 1:
                results += t
        return results, (i1, i2)
    
    def get_color_str(self, i):
        return "{} ({})".format(self.cm[i], '-'.join(map(lambda a: str(a), self.get_tuple(i))))

    
    def breeding_guide(self, cross_slots, clone_slots, store_slots, target_count):
        pool = {}
        store_indicies = {}
        
        i_cross_slots = list(map(lambda a: None if a is None else (self.get_index(a[0]), self.get_index(a[1])), cross_slots))
        i_clone_slots = list(map(lambda a: None if a is None else self.get_index(a), clone_slots))
        i_store_slots = list(map(lambda a: (self.get_index(a[1]), a[2]), store_slots))
        for (store_index, gtype, _) in store_slots: 
            store_indicies[self.get_index(gtype)] = store_index
        
        def add_to_pool(cindex, count=1):
            if cindex not in pool:
                pool[cindex] = 0
            pool[cindex] += count
        
        for slot in i_cross_slots:
            if slot is not None:
                add_to_pool(slot[0])
                add_to_pool(slot[1])
                
        for slot in i_clone_slots:
            if slot is not None:
                add_to_pool(slot)

        for (cindex, count) in i_store_slots:
            add_to_pool(cindex, count)
        
        clones_needed = {}
        for k, v in pool.items():
            if v < target_count:
                clones[k] = target_count - v
                
        all_color_indicies = list(pool.keys())
        new_breeds = []
        for i in range(len(all_color_indicies)):
            for j in range(i + 1):
                c_results, sorted_parents = self.mate_definite(all_color_indicies[i], all_color_indicies[j])
                for c_result in c_results:
                    (c_index, pp) = c_result
                    if c_index not in pool:
                        new_breeds.append((pp, c_index, sorted_parents))
        new_breeds.sort(reverse=True)        
        new_definite_breeds = {} # breeds with 100%, so only one needed
        
        n_cross_slots = [(None, None)] * len(i_cross_slots)
        n_clone_slots = [(None, None)] * len(i_clone_slots)
        
        def try_take_from_pool(parents):
            if parents[0] == parents[1]:
                if pool[parents[0]] >= 2:
                    pool[parents[0]] -= 2
                    return True
            elif pool[parents[0]] >= 1 and pool[parents[1]] >= 1:
                pool[parents[0]] -= 1
                pool[parents[1]] -= 1
                return True
            return False
        
        def try_take_from_pool_single(p):
            if pool[p] >= 1:
                pool[p] -= 1
                return True
            return False
                        
        # cross slot
        for new_breed in new_breeds:
            (pp, c_index, parents) = new_breed
            if c_index in new_definite_breeds:
                continue
            
            # match existing cross_breeds
            materialized = False
            for i, cross_slot in enumerate(i_cross_slots):
                if parents == cross_slot:
                    if try_take_from_pool(parents):
                        n_cross_slots[i] = (parents, None)
                        materialized = True
                        break
            
            # find an empty slot
            if not materialized:
                for i, cross_slot in enumerate(i_cross_slots):
                    if cross_slot is not None:
                        continue
                    if try_take_from_pool(parents):
                        n_cross_slots[i] = (parents, ('+', parents))
                        materialized = True
                        break
            
            if materialized:
                if pp == 1.0:
                    new_definite_breeds[c_index] = True
                    
        # remove from old cross_slot
        for i, oslot in enumerate(i_cross_slots):
            if oslot is None:
                continue
            nslot = n_cross_slots[i]
            if nslot is None:
                n_cross_slots[i] = (None, ('-', oslot))
        
        # clone slot
        for k, v in clones_needed.items():
            # match existing clone_slots
            for i, clone_slot in enumerate(i_clone_slots):
                if v <= 0:
                    break
                if clone_slot == k:
                    if try_take_from_pool_single(k):
                        v -= 1
                        n_clone_slots[i] = (k, None)
                        continue
                        
            # match empty clone_slots
            for i, clone_slot in enumerate(i_clone_slots):
                if v <= 0:
                    break
                if clone_slot is None:
                    if try_take_from_pool_single(k):
                        v -= 1
                        n_clone_slots[i] = (k, ('+', k))
                        continue
        # remove from old clone slot
        for i, oslot in enumerate(i_clone_slots):
            if oslot is None:
                continue
            nslot = n_clone_slots[i]
            if nslot is None:
                n_clone_slots[i] = (None, ('-', oslot))
        
        n_store_slots = [None] * len(i_store_slots)
        for i, (cindex, count) in enumerate(i_store_slots):
            n_store_slots[i] = (cindex, pool[cindex])
            pool[cindex] = 0
        for cindex, c in pool.items():
            if c > 0:
                n_storage_slots.append((cindex, c))
        return n_cross_slots, n_clone_slots, n_store_slots
    
    def print_breeding_guide(self, cross_slots, clone_slots, storage_slots, target_count = 1):
        (n_cross_slots, n_clone_slots, n_store_slots) = self.breeding_guide(cross_slots, clone_slots, storage_slots, target_count)
        def get_pair_str(nsv):
            return 'E' if nsv is None else '{} x {}'.format(self.get_color_str(nsv[0]), self.get_color_str(nsv[1]))
        
        print(self.name)
        print("=====================")
        print("Cross breeding slots:")
        print("---------------------")
        for i, (nsv, delta) in enumerate(n_cross_slots):
            print("{}: {}".format(i+1, get_pair_str(nsv)))
            if delta is not None:
                (op, v) = delta
                print ("\t{} {}".format(op, get_pair_str(v)))
            if nsv is not None:
                for (c, p) in self.mate_definite(nsv[0], nsv[1])[0]:
                    print("\t{} [{}%]".format(self.get_color_str(c), p * 100))
        print("---------------------")                   
        print("Clone slots:")
        print("---------------------")
        for i, (nsv, delta) in enumerate(n_clone_slots):
            print("{}: {}".format(i+1, 'E' if nsv is None else self.get_color_str(nsv)))
            if delta is not None:
                (op, v) = delta
                print ("\t{} {}".format(op, self.get_color_str(v)))
        print( n_store_slots)
        return
        for gtype in l:
            cindex = self.get_index(gtype)
            print("{}: {}".format(pool[cindex], self.get_g_color_str(cindex)))
        print("-----")
        for entry in data:
            (cindex, (p1, p2), pp) = entry
            print("{}: {}: {}[{}] x {}[{}] [{}%]".format(self.get_g_color_str(cindex), self.name, self.get_g_color_str(p1), pool[p1], self.get_g_color_str(p2), pool[p2], pp * 100))
    