---
# 文章标题
title: ACM Try -- 01
# 设置写作时间
date: 2024-01-23
# 一个页面可以有多个分类
category:
  - 算法与数据结构
# 一个页面可以有多个标签
tag:
  - 算法
# 此页面会在文章列表置顶
sticky: true
# 此页面会出现在文章收藏中
star: true
# 侧边栏的顺序
# 数字越小越靠前，支持非整数和负数，比如 -10 < -9.5 < 3.2, order 为 -10 的文章会最靠上。
# 个人偏好将非干货或随想短文的 order 设置在 -0.01 到 -0.99，将干货类长文的 order 设置在 -1 到负无穷。每次新增文章都会在上一篇的基础上递减 order 值。
order: -0.002
--- 

## 1. Dragon String (龙字符串)

### 题目描述
给定一个正整数N，要求输出一个特定的字符串，该字符串由一个'L'、N个'o'、一个'n'和一个'g'按顺序组成。

### 约束条件
- 1 ≤ N ≤ 2024
- N为整数

### 解题思路
这是一道简单的字符串构造题，主要考察基础的字符串操作和循环控制。解题步骤如下：
1. 读取输入整数N
2. 输出字符'L'
3. 循环N次输出字符'o'
4. 最后输出字符'n'和'g'

### 易错点
1. 输出字符的大小写问题：'L'必须是大写，'o'、'n'、'g'必须是小写
2. 循环次数控制：容易多输出或少输出一个'o'
3. 输出顺序：必须严格按照L->o->n->g的顺序

### 代码实现
```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    cin >> n;
    cout << 'L';
    while (n-- > 0) {
        cout << 'o';
    }
    cout << 'n' << 'g';
    return 0;
}
```

### 复杂度分析
- 时间复杂度：O(N)
- 空间复杂度：O(1)

## 2. YES字符串判断

### 题目描述
给定T个测试用例，每个用例包含一个字符串，判断该字符串是否等于"YES"（不区分大小写）。

### 约束条件
- 1 ≤ T ≤ 103
- 输入字符串只包含大小写英文字母

### 解题思路
这是一道字符串比较题，主要考察字符串处理和大小写转换。解题要点：
1. 将输入字符串转换为小写
2. 与标准字符串"yes"比较
3. 注意处理大小写不敏感的比较

### 易错点
1. 输入字符串长度未判断：虽然题目保证输入合法，但在实际工程中应该加上长度判断
2. 输出大小写问题：输出"YES"或"NO"时必须全部大写
3. 字符串比较前的转换：必须先将输入字符串转换为小写，再进行比较
4. 返回值处理：main函数末尾漏掉return 0

### 代码实现
```cpp
#include <bits/stdc++.h>
#include <cctype>
#include <cstdio>
using namespace std;

int main() {
    int n;
    cin >> n;
    string boolstr;
    while (n -- > 0) {
        cin >> boolstr;
        // 将整个字符串转换为小写
        for(char &c : boolstr) {
            c = tolower(c);
        }
        if (boolstr == "yes") {
            printf("YES\n");
        } else {
            printf("NO\n");
        }
    }
}
```

### 复杂度分析
- 时间复杂度：O(T×L)，其中L为字符串长度
- 空间复杂度：O(1)

## 3. 奇偶判断

### 题目描述
给定N个大整数，判断每个数的奇偶性。

### 约束条件
- 1 ≤ N ≤ 100
- 输入整数不超过10^60

### 解题思路
这是一道大数处理题，但有一个重要的数学性质：一个数的奇偶性只取决于其最后一位数字。因此：
1. 将输入数字以字符串形式读入
2. 只需判断最后一个字符对应的数字的奇偶性
3. 无需进行完整的大数运算

### 易错点
1. 大数处理方式：试图将输入转换为整数类型会导致溢出
2. 字符转数字：忘记将字符转换为数字（减去'0'）就直接判断
3. 输出格式：输出"odd"或"even"时必须全部小写
4. 边界情况：没有考虑输入为单个数字的情况

### 代码实现
```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    cin >> n;
    string num;
    while (n-- > 0) {
        cin >> num;
        // 只需要检查最后一位数字即可判断奇偶性
        if ((num.back() - '0') % 2 == 1)
            cout << "odd\n";
        else
            cout << "even\n";
    }
    return 0;
}
```

### 复杂度分析
- 时间复杂度：O(N)
- 空间复杂度：O(1)

## 4. 字符统计问题

### 题目描述
给定一个字符串和目标次数m，需要计算使每个字符出现次数达到m次所需添加的最少字符数。

### 解题思路
这是一道哈希统计题，主要思路如下：
1. 使用哈希数组统计每个字符的出现次数
2. 对于每个字符，计算需要补充多少个才能达到目标次数m
3. 累加所有需要补充的字符数

### 易错点
1. 哈希数组大小：vector初始化大小必须足够存储所有可能的字符
2. 字符映射：将字符转换为数组下标时的计算可能越界
3. 补充计算：当字符出现次数已经超过m时，不需要再补充
4. 测试用例处理：忘记处理多组测试用例的情况

### 代码实现
```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int cnt;
    cin >> cnt;
    while (cnt -- > 0) {
        int m, n;
        int count = 0;
        cin >> n >> m;
        string line;
        vector<int> hash(7);
        cin >> line;
        for (auto &ch : line) {
            hash[Hash(int(ch - 'A'))]++;
        }
        for (auto k : hash) {
            if (m > k)
                count += m - k;
        }
        printf("%d\n", count);
    }
    return 0;
}
```

### 复杂度分析
- 时间复杂度：O(T×N)，其中T为测试用例数，N为字符串长度
- 空间复杂度：O(1)，因为哈希数组大小固定

## 5. 投票统计

### 题目描述
给定n个人在m天内对规则k的投票情况，判断规则k是否符合民意。规则符合民意的条件是：在过半数的天数中，有过半数的人支持该规则。

### 解题思路
这是一道二维统计题，需要两层判断：
1. 对每一天统计支持规则k的人数，判断是否过半
2. 统计符合条件的天数是否过半
3. 使用排序来简化判断过半的操作

### 易错点
1. 过半数计算：(n+1)/2 而不是 n/2，需要考虑奇数情况
2. 数组初始化：right数组的大小应该是m而不是当前的m值（因为m在循环中会减少）
3. 排序后的判断：需要判断right[0]和right[(days-1)/2]，而不是简单统计1的个数
4. 变量保存：需要保存原始的m值用于最后判断

### 代码实现
```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, m, k;
    cin >> n >> m >> k;
    vector<int> right(m, 0);
    int days = m;
    while (m -- > 0) {
        int cnt = 0;
        int num = 0;
        for (int i = 0; i < n; i++) {
            cin >> num;
            if (num == k)
                cnt++;
        }
        if (cnt >= (n + 1) / 2) {
            right[m] = 1;
        }
    }
    sort(right.begin(), right.end(), [](int x, int y){return x > y;});
    if (right[0] == 1 && right[(days - 1) / 2] == 1)
        printf("YES");
    else
        printf("NO");
    return 0;
}
```

### 复杂度分析
- 时间复杂度：O(M×N + MlogM)
- 空间复杂度：O(M)

## 6. 字符替换

### 题目描述
给定一个字符串S和Q次操作，每次操作将字符串中的某个字符全部替换为另一个字符。

### 解题思路
这是一道字符串模拟题，主要考察字符串的遍历和替换操作：
1. 读入初始字符串
2. 对每次操作，遍历整个字符串进行替换
3. 注意替换操作要同时进行，避免连锁反应

### 易错点
1. 替换顺序：每次操作必须同时替换所有匹配的字符，不能边替换边比较
2. 自身替换：当c和d相同时也需要正确处理
3. 字符串修改：使用引用或指针修改字符串时的正确性
4. 输出格式：最后需要输出换行符

### 代码实现
```cpp
#include <iostream>
#include <string>
using namespace std;

int main() {
    int N;
    string S;
    int Q;
    cin >> N >> S >> Q;
    for(int i = 0; i < Q; i++) {
        char c, d;
        cin >> c >> d;
        for(int j = 0; j < N; j++) {
            if(S[j] == c) {
                S[j] = d;
            }
        }
    }
    cout << S << endl;
    return 0;
}
```

### 复杂度分析
- 时间复杂度：O(Q×N)
- 空间复杂度：O(N)

## 7. 矩阵操作

### 题目描述
给定一个n×n的矩阵和m次操作，每次操作可以交换两行或两列。

### 解题思路
这是一道矩阵模拟题，但有一个优化技巧：
1. 不直接修改矩阵，而是维护行列的映射关系
2. 每次交换只需要修改映射数组
3. 最后输出时根据映射关系重建矩阵

### 易错点
1. 下标转换：输入的行列号从1开始，需要减1后再使用
2. 映射数组初始化：必须正确初始化row和col数组为0到n-1
3. 矩阵访问：输出时使用映射数组访问matrix[row[i]][col[j]]而不是直接访问
4. 输出格式：最后一个数字后不能有空格，需要换行

### 代码实现
```cpp
#include <iostream>
#include <vector>
using namespace std;

int main() {
    int n, m;
    cin >> n >> m;
    vector<vector<int>> matrix(n, vector<int>(n));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cin >> matrix[i][j];
        }
    }
    vector<int> row(n), col(n);
    for (int i = 0; i < n; i++) {
        row[i] = i;
        col[i] = i;
    }
    for (int i = 0; i < m; i++) {
        int op, x, y;
        cin >> op >> x >> y;
        x--; y--;
        if (op == 1) {
            swap(row[x], row[y]);
        } else {
            swap(col[x], col[y]);
        }
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cout << matrix[row[i]][col[j]] << (j == n-1 ? '\n' : ' ');
        }
    }
    return 0;
}
```

### 复杂度分析
- 时间复杂度：O(n² + m)
- 空间复杂度：O(n²)

## 总结

这组题目涵盖了多个基础算法知识点：
1. 字符串处理（题目1、2、6）
2. 大数处理技巧（题目3）
3. 哈希统计（题目4）
4. 二维统计与排序（题目5）
5. 矩阵操作与状态维护（题目7）

每道题都体现了一些重要的编程思想：
- 善用数学性质简化问题（题目3）
- 使用哈希表优化统计（题目4）
- 排序简化判断（题目5）
- 状态映射代替直接修改（题目7）

这些思想在更复杂的算法题中都会经常用到。 
